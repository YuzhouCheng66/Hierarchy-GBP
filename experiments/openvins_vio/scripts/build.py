#!/usr/bin/env python3
"""Prepare isolated pinned sources and build clean and integrated executables."""

from pathlib import Path
import argparse, datetime, hashlib, json, os, subprocess
from patch_openvins import apply, HEAD

PACKAGE = Path(__file__).resolve().parents[1]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--work-dir", type=Path, required=True)
    p.add_argument("--reference-repo", type=Path)
    p.add_argument("--dependency-prefix", type=Path)
    p.add_argument("--mkl-lib-dir", type=Path, required=True)
    p.add_argument("--jobs", type=int, default=3)
    p.add_argument("--only", choices=("both", "original", "integrated"), default="both")
    args = p.parse_args()
    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    reference = (
        args.reference_repo.resolve()
        if args.reference_repo
        else work / "openvins-reference"
    )
    if not reference.exists():
        subprocess.run(
            ["git", "clone", "https://github.com/rpng/open_vins.git", str(reference)],
            check=True,
        )
    for mode in ("original", "integrated"):
        if args.only not in ("both", mode):
            continue
        source = work / ("source-" + mode)
        if not source.exists():
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(reference),
                    "worktree",
                    "add",
                    "--detach",
                    str(source),
                    HEAD,
                ],
                check=True,
            )
        assert (
            subprocess.check_output(
                ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
            ).strip()
            == HEAD
        )
        old_manifest = work / ("build-" + mode) / "build-manifest.json"
        previous = (
            json.loads(old_manifest.read_text()).get("patch")
            if old_manifest.exists()
            else None
        )
        patch = apply(source, previous) if mode == "integrated" else None
        if mode == "original":
            assert not subprocess.check_output(
                ["git", "-C", str(source), "status", "--porcelain"], text=True
            )
        build = work / ("build-" + mode)
        build.mkdir(exist_ok=True)
        record_path = build / "build-manifest.json"
        if record_path.exists():
            previous = build / (
                "build-manifest-"
                + datetime.datetime.now().strftime("%Y%m%dT%H%M%S%f")
                + ".json"
            )
            previous.write_bytes(record_path.read_bytes())
        paths = ["/usr/lib/x86_64-linux-gnu"]
        if args.dependency_prefix:
            paths += [
                str(args.dependency_prefix / "lib"),
                str(args.dependency_prefix / "lib/x86_64-linux-gnu"),
            ]
        paths += [str(args.mkl_lib_dir)]
        env = dict(
            os.environ,
            LD_LIBRARY_PATH=":".join(paths),
            OMP_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            MKL_DYNAMIC="FALSE",
            MKL_THREADING_LAYER="SEQUENTIAL",
            MKL_INTERFACE_LAYER="LP64",
        )
        cmd = [
            "cmake",
            "-S",
            str(source / "ov_msckf"),
            "-B",
            str(build),
            "-G",
            "Ninja",
            "-DENABLE_ROS=OFF",
            "-DENABLE_ARUCO_TAGS=OFF",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG -g0 -march=native -DEIGEN_DONT_PARALLELIZE",
            "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
            "-DEIGEN3_INCLUDE_DIR=/usr/include/eigen3",
            "-DOV_GAUSSIAN_PACKAGE=" + str(PACKAGE),
            "-DOV_GAUSSIAN_INTEGRATION=" + ("ON" if patch else "OFF"),
            "-DOV_MKL_LIB_DIR=" + str(args.mkl_lib_dir),
            "-DCMAKE_PROJECT_INCLUDE=" + str(PACKAGE / "cmake/hook.cmake"),
            "-DCMAKE_BUILD_RPATH=" + ";".join(paths),
        ]
        if args.dependency_prefix:
            cmd += [
                "-DCMAKE_PREFIX_PATH=" + str(args.dependency_prefix),
                "-DCMAKE_MODULE_PATH="
                + str(args.dependency_prefix / "share/glog/cmake"),
                "-DCMAKE_CXX_FLAGS=-I" + str(args.dependency_prefix / "include"),
                "-DCMAKE_EXE_LINKER_FLAGS=-Wl,-rpath-link,"
                + str(args.dependency_prefix / "lib"),
            ]
        record = dict(
            schema="live_openvins_build_v1",
            mode=mode,
            started_utc=now(),
            upstream_commit=HEAD,
            patch=patch,
            configure=cmd,
            package_source_sha256={
                x.relative_to(PACKAGE).as_posix(): sha(x)
                for x in PACKAGE.rglob("*")
                if x.is_file()
                and "__pycache__" not in x.parts
                and "results" not in x.parts
            },
            environment={
                k: env[k]
                for k in (
                    "LD_LIBRARY_PATH",
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "MKL_DYNAMIC",
                    "MKL_THREADING_LAYER",
                    "MKL_INTERFACE_LAYER",
                )
            },
        )
        log = build / (
            "build-" + datetime.datetime.now().strftime("%Y%m%dT%H%M%S") + ".log"
        )
        with log.open("w") as f:
            r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
            record["configure_returncode"] = r.returncode
            if not r.returncode:
                targets = ["run_euroc_folder"] + (
                    ["test_gaussian_backend"] if patch else []
                )
                compile = [
                    "cmake",
                    "--build",
                    str(build),
                    "--target",
                    *targets,
                    "-j",
                    str(args.jobs),
                ]
                record["compile"] = compile
                r = subprocess.run(compile, stdout=f, stderr=subprocess.STDOUT, env=env)
                record["build_returncode"] = r.returncode
        record["finished_utc"] = now()
        record["log"] = log.name
        record["source_diff"] = subprocess.check_output(
            ["git", "-C", str(source), "diff", "--stat"], text=True
        )
        if not r.returncode:
            record["binary_sha256"] = {
                n: sha(build / n)
                for n in ["run_euroc_folder", "libov_msckf_lib.so"]
                + (
                    ["test_gaussian_backend", "libov_gaussian_backend.a"]
                    if patch
                    else []
                )
            }
            record["compile_commands_sha256"] = sha(build / "compile_commands.json")
        record_path.write_text(json.dumps(record, indent=2) + "\n")
        print(mode, "build", r.returncode, flush=True)
        if r.returncode:
            print(log.read_text()[-7000:])
            raise SystemExit(r.returncode)


if __name__ == "__main__":
    main()
