# SE2 CUDA H-GBP: strict CPU16 comparison

This is the retained `transportsegment` hybrid implementation, not the old
single-linearization, identity-basis prototype. The latter's historical 5.11x
claim was not an equivalent-algorithm comparison and must not be cited.
Its source remains available in Git history before this directory replaced it.

**Status: numerical validation passes; the complete-process 2x target is NOT met.**
The GPU accelerates lower GBP. Conditioned basis, nonlinear relinearization,
coarse assembly/restriction/prolongation and sparse coarse solve remain on CPU.
The coarse solver is Eigen `SimplicialLDLT`, not CHOLMOD in this selected path.
No experimental incoming-storage or other unproven kernel is selected here.

## Two different timing boundaries

The original reported checkpoint (2026-09-28, nine interleaved repeats after
one excluded warm-up, milliseconds) is:

| Dataset | CPU16 solver | GPU/hybrid solver | Solver speedup | CPU16 process wall | GPU process wall | Process speedup |
|---|---:|---:|---:|---:|---:|---:|
| FR079 | 109.42 | 66.02 | 1.66x | 238.70 | 266.88 | 0.89x |
| FRH | 124.74 | 71.41 | 1.75x | 288.01 | 300.06 | 0.96x |
| M3500 | 282.93 | 143.54 | 1.97x | 588.69 | 519.85 | 1.13x |

Solver time includes workspace allocation, transfers, all 20 outers, basis,
coarse corrections and objectives. It excludes CUDA context initialization,
input parsing, output serialization and process teardown. Process wall is timed
by the parent across launch, output capture and exit, with the same input and
full per-outer pose JSON output for CPU and GPU. CUDA context costs about
57-59 ms and is **already included in GPU process wall**; never add it twice.
These are fresh processes with warmed filesystem/driver caches, not first-boot
machine measurements. No persistent server or pre-initialized context is hidden.

An independent build of this packaged directory was tested again:

| Dataset | CPU16 solver | GPU/hybrid solver | CPU16 process wall | GPU process wall |
|---|---:|---:|---:|---:|
| FR079 | 122.33 | 67.57 | 260.42 | 281.37 |
| FRH | 133.02 | 73.06 | 304.58 | 315.93 |
| M3500 | 289.23 | 149.05 | 602.57 | 529.89 |

This is packaging revalidation, not a new optimization gain. CPU and GPU clock
variation changes ratios. Neither cohort meets 2x complete-process acceleration.
Hardware: i9-13900HX, CPU affinity `0xFFFF`, 16 OpenMP threads; RTX 4080 Laptop
(58 SM), CUDA 12.8.61, MSVC 19.44, NVIDIA driver 581.42. Sixteen threads does
not imply sixteen physical cores.

## Algorithm and execution

- Use the shared release policy in `../../../benchmarks/configs/pgo.json`: outer20, c3, k100,
  group size20, reduced rank4, Huber5, FP64, jitter1e-7, mode2 initialization.
- Rebuild the same conditioned basis every outer. Do not substitute identity
  basis, reduce sweep budgets, relax precision gates or add a fine direct solve.
- Build degree/slot ownership once. Power-of-two source tiles use 2-32 lanes
  with exact ordered incoming sums and cached incoming indices.
- Run one cooperative persistent kernel per 100-sweep batch, 32 threads/CTA.
  Full-lambda updates use CPU-matched FP64 arithmetic. Adaptive convergence
  retains the same check/freeze/thaw schedule and then uses cached eta maps.
- Replay a CUDA Graph containing packed upload, lower sweeps and result
  download. Retain device allocations across cycles; this is NOT an entirely
  GPU-resident nonlinear solver. The first post-coarse sweep consumes the
  actual injected belief, not a reconstructed stale belief.
- Overlap CPU basis construction with the first GPU batch using a snapshot
  of the correct pre-initialization boundary precision. Cache the coarse sparse
  pattern without reusing stale numerical matrix values.

| Dataset | Full-lambda sweeps | Eta-only sweeps | Coarse solves | Final CPU and GPU cost |
|---|---:|---:|---:|---:|
| FR079 | 6000 | 0 | 60 | 18.785887502887988 |
| FRH | 1100 | 4900 | 60 | 0.00009656841046587548 |
| M3500 | 2036 | 3964 | 60 | 68.95754667615267 |

Source is frozen from the validated segment/transport checkpoint, with line
endings normalized. The CUDA entry defines only the retained configuration;
inactive source guards are retained conservatively rather than changing kernel
arithmetic during packaging. CPU kernels and numerical policies are unchanged;
build paths follow the current repository layout. Archived records retain their
original paths. Release reference: `6276e2c731034951581dbc6902c1f4d2e0159bdc`.

## Build and reproduce

Use Windows x64, Visual Studio 2022 Build Tools, CMake/Ninja, CUDA 12.8 and
the repository's Eigen/SuiteSparse/OpenBLAS dependencies. DependencyRoot must
contain `external/eigen` and `vcpkg_installed/x64-windows`; RuntimeDir supplies
the same validated DLL set used by the CPU release. No executables, vendor
libraries or datasets are committed here.

From this directory:

```powershell
.\build.ps1 -DependencyRoot C:\deps\hgbp -RuntimeDir C:\deps\hgbp-runtime -Jobs 8
ctest --test-dir build --output-on-failure
python -m unittest discover -p 'test_*.py'
.\run.ps1 -DataDirectory C:\data\se2 -Shadow -Repeats 0 -OutputDirectory results/shadow
.\run.ps1 -DataDirectory C:\data\se2 -Repeats 9 -OutputDirectory results/timing
python compare_resident.py results/timing --shadow-directory results/shadow
```

Input names are `FR079.g2o`, `FRH.g2o`, `M3500.g2o`; the runner rejects hashes
different from `../../../benchmarks/configs/datasets.json`. See the repository data download
instructions. It also rejects CPU source drift from the pinned release.
`repeat=0` is excluded; subsequent CPU/GPU order alternates. Never run a build,
profiler or another benchmark concurrently with timing. The output directory
must not exist. Run the script in a child PowerShell if its controlled process
environment must not affect an interactive shell.

To compare with the production CPU executable rather than the CPU target from
this independent build, pass `-ReferenceBinary <main-build>/pgo/se2_benchmark.exe`
to both shadow and timing commands. The override is explicit, dataset-independent
and recorded with the executable hash. It changes only the reference executable,
not any numerical policy. The latest [cleanup revalidation](../../docs/recertification.md)
uses the production reference and nine timing pairs per dataset.

The current runner pins numerical commit
`8881978244a7a46cb75445d57bf617c17546077b`, checking the cleaned CPU tree (including
private `.inc` files), CUDA sources and build settings. Historical evidence retains
its original locks. The runner hash and numerical reference are recorded separately;
updating a layout does not bypass numerical equivalence checks.

Sanitizers and Nsight are separate diagnostics, never speedup samples:

```powershell
.\run.ps1 -DataDirectory C:\data\se2 -Variants transportsegment -Repeats 0 `
  -Sanitizer memcheck -OutputDirectory results/memcheck
.\run.ps1 -DataDirectory C:\data\se2 -Variants transportsegment -Repeats 0 `
  -NsightSystems 'C:\path\to\nsys.exe' -NsightGraphTrace node `
  -NsightDurationSeconds 10 -OutputDirectory results/nsight
```

## Evidence and acceptance

[RESULTS.md](RESULTS.md) contains phase accounting and remaining bottlenecks.
`evidence/reported_20260928` is the exact previously reported cohort;
`evidence/package_20260928` is the independent package build. Each has a
machine-readable summary and losslessly deduplicated raw result, pose and log
files. Archives retain all repeats and warm-ups, executable/input hashes,
per-outer policies, costs and poses; no absolute workstation paths are needed.

```powershell
python archive_results.py --verify evidence/reported_20260928
python archive_results.py --verify evidence/package_20260928
```

All recorded CPU/GPU outer costs and poses match exactly in these cohorts.
All 60 batch shadow comparisons pass for messages, beliefs and means; adaptive
freeze/thaw policies match. Native tests cover packet layout, overlap callbacks,
coarse pattern equivalence, non-SPD failure rollback and subsequent recovery.
Shadow/sanitizer timings are excluded. `accepted_2x` in the compatibility summary
means solver-only; **`accepted_wall_2x` is the complete-process acceptance gate**.
Numerical certification is necessary but not evidence of 2x acceleration.
