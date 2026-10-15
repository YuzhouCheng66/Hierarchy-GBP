# Dependencies and runtime

The validated build uses MSVC 14.44, AVX2, Eigen 3.4.0, SuiteSparse Config
7.8.3 / CHOLMOD 5.3.0, and OpenBLAS 0.3.29. The CMake project preserves the
validated floating-point options: SE2's residual kernel uses `/fp:fast`;
SE3 does not. Other platforms/toolchains have not been release-validated.

```text
<dependency-root>/
  external/eigen/
  vcpkg_installed/x64-windows/{include,lib,bin}/
  external_baselines/rootba/
    src/ and external/
    build-win/src/rootba/Release/
    conda_env/Library/{include,lib,bin}/
```

RootBA reference revision: `d3900037fc4bc98328e5d8d26f1c4eed922f88cc`.
BA reuses its BAL data handling and linearization support, not its linear
solver as a substitute for H-GBP.

PGO requires a shared OpenBLAS **LP64** runtime exporting the LAPACKE
`dsbevx_work`, `dsbtrd_work`, and `dstemr_work` functions. A minimal BLAS-only
DLL is not interchangeable. The validated runtime reports
`OpenBLAS 0.3.29 DYNAMIC_ARCH NO_AFFINITY Haswell MAX_THREADS=64` and its
`openblas.dll` SHA-256 is
`3fdccaf0054074f5979613a883b094db90a57395dc7ea31d5fd818c226ba2a1f`.

The official [OpenBLAS 0.3.29 Windows x64 archive](https://github.com/OpenMathLib/OpenBLAS/releases/download/v0.3.29/OpenBLAS-0.3.29_x64.zip)
provides an LP64 build. Do not use the `_x64_64` (ILP64) variant. Check the
runtime exports and numerical tests when substituting a different build.

Pass a directory containing OpenBLAS and its companion DLLs using
`-PGORuntimeDir` or `HGBP_PGO_RUNTIME`. `code/cmake/openblas_banded.def` builds
the Windows import library. The script stages vcpkg's SuiteSparse/LAPACK
dependencies first, then this OpenBLAS runtime into `build/pgo`.
BA receives its TBB/fmt/glog/gflags runtime dependencies plus the vcpkg
numerical DLLs in `build/ba`; it does not shadow the system MSVC/OpenMP DLLs
with conda copies. The build does not modify the dependency installations.

For explicit locations, the build script accepts `-EigenDir`,
`-VcpkgInstalledDir` and `-RootBADir`. PGO-only users may omit RootBA with
`-PGOOnly` (CMake: `-DHGBP_BUILD_BA=OFF`).

RootBA source, build and installed dependencies need not share a directory.
After preparing the pinned checkout and Release library using the
[BA dependency recipe](../../benchmarks/baselines/ba/README.md#source-and-build),
pass the actual library and prefix explicitly; no copying into `build-win` or
`conda_env` is required:

```powershell
.\code\build.ps1 -DependencyRoot C:\deps\hgbp -PGORuntimeDir C:\deps\pgo-runtime `
  -RootBADir C:\deps\rootba -RootBALibrary C:\deps\rootba\build\src\rootba\Release\rootba.lib `
  -RootBAPrefix C:\deps\rootba-runtime
```

The prefix contains matching `include/`, `lib/` and `bin/` directories for
TBB (`tbb12`), fmt, glog and gflags. The library path must name the actual output
of your generator; Ninja may omit the `Release` component. CMake equivalents
are `HGBP_ROOTBA_LIBRARY` and `HGBP_ROOTBA_PREFIX`; the original paths above
remain the defaults. Keep compiler, CRT, Eigen and dependency versions matched
to the pinned build; a different installation is not automatically certified.

Use `-CheckOnly` with the same arguments to check required files and tool
availability without creating a build directory, installing packages or changing
dependency trees. Missing libraries are listed before compilation. This is a
preflight check, not a numerical or binary-compatibility certification; run the
build and tests afterward. Main and CUDA build scripts share the x64 MSVC setup
in `cmake/msvc.ps1`; the optional CUDA build still requires its own validation.

The [cleanup revalidation](recertification.md) documents the matching BA
dependency recovery, full BA rerun and separate native SE2 CUDA certification.
Its package hashes describe the tested environment, not a requirement to use
the original workstation's absolute paths.

Dataset checksums and preprocessing are specified in `benchmarks/configs/datasets.json`.
Absolute paths in local validation records are not required by the release.
