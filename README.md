# Hierarchy-GBP: Accelerating Factor Graph Inference via Abstraction and Recovery

H-GBP accelerates **CPU bundle adjustment and pose graph optimization** through
loopy Gaussian message passing and coarse-scale correction. Optimized kernels,
packed messages and cached updates support single-threaded and parallel execution.

[**Project**](https://yuzhoucheng.com/Hierarchy-GBP/) |
[**Paper**](https://arxiv.org/abs/2610.06978) |
[**Data**](https://huggingface.co/datasets/yuzhoucheng66/HGBP-Benchmarks)

[![H-GBP mechanism: GBP smooths high-frequency error; abstraction, correction and recovery reduce low-frequency error.](https://raw.githubusercontent.com/YuzhouCheng66/Hierarchy-GBP/8c88204504344d255ca12e26211200908c97b7cb/code/docs/assets/teaser.gif)](https://yuzhoucheng.com/Hierarchy-GBP/)

## Layout

```text
code/          Solvers, numerical tests and optional variants
benchmarks/    Shared policies, datasets and baseline reproduction
integrations/  BAL/g2o model interfaces and COLMAP input conversion
site/          Project page, animations and published results
```

Start with the [solver entry points](code/docs/architecture.md).
Policies are shared across datasets within SE2, SE3 and BA respectively;
dataset names do not select numerical overrides. `main` is maintained;
`legacy` preserves the preceding main version.

## Build

Validated on Windows x64 with Visual Studio 2022, CMake/Ninja, Python 3.10+,
Eigen, SuiteSparse and OpenBLAS. See [H-GBP build dependencies](code/docs/dependencies.md)
for setup. Running comparison baselines is optional.

```powershell
.\code\build.ps1 -DependencyRoot C:\deps -PGORuntimeDir C:\deps\pgo-runtime
ctest --test-dir build --output-on-failure
```

Use `-PGOOnly` to omit BA. Separate `build/pgo` and `build/ba` runtime
directories prevent conflicting BLAS libraries from being mixed.

## Data & Reproduction

The [Hugging Face release](https://huggingface.co/datasets/yuzhoucheng66/HGBP-Benchmarks)
contains **9 pose graphs** and **10 BA problems**. The downloader verifies
pinned SHA-256 checksums and restores `data/pgo` and `data/ba`.

```powershell
python benchmarks/download.py
python benchmarks/run.py ba --threads 16 --output-root results/reproduction/ba
python benchmarks/run.py pgo --threads 16 --output-root results/reproduction/pgo
```

Append dataset names for a subset; use `--threads 1` or `--compare-threads`
for serial runs or sequential 1/16-thread comparison. Run `--help` for custom
inputs and runtime locations. Outputs include commands, hashes, native solver
and process wall times, final cost, and BA reprojection metrics.

See [algorithm policies](code/docs/policies.md),
[baseline reproduction](benchmarks/baselines/README.md),
[Globe generators](benchmarks/datasets/README.md) and
[validation](code/docs/recertification.md).

## Local Message Passing at Every Level

The optional [`pure_gbp`](code/variants/pure_gbp/README.md) variant uses GBP on
the fine graph and both coarse levels, without a coarse direct or Krylov solve.
Paired 16-thread, outer20 tests on nine PGO inputs take **1.45-2.51x** the
default runtime, with at most **0.451%** higher final cost. The coarse-direct
route remains the default; [code and evidence](code/variants/pure_gbp/README.md)
are isolated from it.

## Interfaces

Use standard **BAL** for BA and **g2o** for SE2/SE3 pose graphs.
The separate [`integrations/`](integrations/README.md) adapters export optimized
models and convert supported COLMAP text reconstructions to BAL. BAL/g2o files
provide interchange with Ceres examples and g2o/GTSAM utilities; these are
file interfaces, not drop-in solver backends. The validated kernels are unchanged.

## Citation

If you find this code/work useful, please consider citing:

```bibtex
@article{cheng2026hierarchy,
  title   = {{Hierarchy-GBP}: Accelerating Factor Graph Inference via Abstraction and Recovery},
  author  = {Cheng, Yuzhou and Yates, Tom and Alzugaray, Ignacio and
             Akarca, Danyal and Mediano, Pedro A. M. and Davison, Andrew J.},
  journal = {arXiv preprint arXiv:2610.06978},
  year    = {2026},
  url     = {https://arxiv.org/abs/2610.06978}
}
```

See [CITATION.cff](CITATION.cff) for citation-manager metadata.
