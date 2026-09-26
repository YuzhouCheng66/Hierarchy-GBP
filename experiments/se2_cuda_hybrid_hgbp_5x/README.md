# SE2 CUDA/CPU Hybrid H-GBP 5x Prototype

This directory preserves the first working SE2 hybrid system that connected a
CUDA Gaussian-belief-propagation smoother to a CPU CHOLMOD coarse solve. On
M3500 it produced the historical `66.67 ms` prototype timing, compared with a
`340.48 ms` 16-thread CPU H-GBP reference (`5.11x`).

## Scope and provenance

The CUDA source is the earliest intact local snapshot that still contains the
measured legacy path. It was recovered from the experiment workspace dated
2026-07-18. The repository copy only normalizes mixed CRLF/LF line endings;
its SHA-256 is:

```text
d5425baff092d82e695e78c2b035f3d15fc3c8f8a38345650a59e05d843f5ec6
```

The original mixed-line-ending snapshot SHA-256 was
`48f5b0739649aa99854c3cc889ba4c8d304827851e6a3b2eb1df374e78e0f7e8`.

The exact pre-snapshot file was not in Git history. This snapshot contains
later optional sparse/SVD experiments as well, but the command below explicitly
selects the original dense identity-group path with `--hybrid-dense-coarse`.

## Pipeline

```text
GPU lower GBP sweeps
  -> GPU dense identity-group coarse A/b assembly
  -> D2H coarse A/b
  -> CPU CHOLMOD factorization and solve
  -> H2D coarse delta
  -> GPU prolongation and belief update
```

The lower solver is genuine Gaussian BP rather than block Jacobi. A persistent
cooperative CUDA kernel alternates factor-to-variable Schur messages and
variable-belief reductions, with grid-wide synchronization between phases.
The M3500 run uses analytic SE2 linearization, exact incoming-message rows,
double-precision canonical messages, a 3x3 inverse Schur kernel, Huber delta 5,
and fixed-lambda/eta-only sweeps after sweep 60.

CHOLMOD is loaded at runtime with `LoadLibrary`; the executable therefore has
no static dependency on `cholmod.dll`. Set `HGBP_CHOLMOD_BIN` to the directory
containing CHOLMOD and its SuiteSparse runtime DLLs.

## Historical M3500 result

| path | group size | coarse dim | cycles | time | ratio | speedup |
|---|---:|---:|---:|---:|---:|---:|
| CPU H-GBP, 16 threads | 20 | `r=4` | `outer=20, c=3` | 340.48 ms | 1.0000 | 1.00x |
| GPU/CPU hybrid prototype | 20 | 525 | 20 | 66.67 ms | 0.1958 | 5.11x |
| GPU/CPU hybrid prototype | 15 | 702 | 20 | 80.23 ms | 0.2356 | 4.24x |

The `group-size=15` timing decomposed as follows:

| phase | time |
|---|---:|
| GPU lower GBP | 33.01 ms |
| GPU coarse assembly | 1.47 ms |
| dense coarse D2H | 7.85 ms |
| CPU CHOLMOD solve | 21.79 ms |
| coarse delta H2D + GPU apply | 0.84 ms |

The fixed-lambda GPU lower result matched the CPU16 lower reference to about
`2.6e-9` relative mean error.

## 2026-09-26 revalidation

The frozen source was rebuilt with CUDA 12.8 and Visual Studio 2022 Build Tools
and rerun on the original RTX 4080 Laptop GPU. The output confirmed
`coarse_mode=dense`, `basis_mode=identity`, `coarse_dim=525`, and 20 cycles.

After one cold DLL/cache run, three fresh process runs measured:

```text
63.496 ms, 68.337 ms, 67.317 ms
median = 67.317 ms
67.317 / 340.48 = 0.19771  (5.06x)
```

The first cold run was `270.716 ms`, of which `209.117 ms` was CHOLMOD DLL
loading. This is why cold and warm measurements must be reported separately.
The lower-only verification measured `2.561337e-9` relative mean error against
the 16-thread CPU reference. Full machine-readable results are in
[`revalidation_2026-09-26.json`](revalidation_2026-09-26.json).

### Important comparison boundary

The `5.11x` number is not a final end-to-end, algorithm-equivalent result. The
GPU measurement runs 20 hybrid cycles on one already-linearized graph with a
3-DOF identity basis per group. The CPU denominator runs 20 nonlinear outers
with a learned `r=4` SVD basis and three inner cycles per outer. This directory
preserves a systems prototype and its mechanism; it must not be cited as a
full nonlinear H-GBP speedup.

## Build

Requirements:

- Windows x64 and Visual Studio 2022
- CUDA Toolkit with compute capability 8.9 support
- Eigen headers
- SuiteSparse headers at compile time
- CHOLMOD and SuiteSparse DLLs at runtime

The direct `nvcc` path is the validated build on the original workstation; it
does not require the Visual Studio CMake CUDA integration:

```powershell
$env:EIGEN3_INCLUDE_DIR = "C:\deps\eigen"
$env:SUITESPARSE_INCLUDE_DIR = "C:\deps\suitesparse\include\suitesparse"
.\build_windows.bat
```

An equivalent CMake configuration is supplied for machines with a registered
Visual Studio CUDA toolset:

```powershell
cmake -S . -B build -G Ninja `
  -DCMAKE_BUILD_TYPE=Release `
  -DEIGEN3_INCLUDE_DIR=C:\deps\eigen `
  -DSUITESPARSE_INCLUDE_DIR=C:\deps\suitesparse\include\suitesparse `
  -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build
```

## Reproduce the historical path

```powershell
.\run_m3500.ps1 `
  -Dataset C:\data\M3500.g2o `
  -CholmodBin C:\deps\suitesparse\bin `
  -GroupSize 20
```

The expanded solver command is:

```powershell
.\build\se2_cuda_hybrid_hgbp.exe `
  --problem-file C:\data\M3500.g2o `
  --hybrid-hgbp `
  --hybrid-dense-coarse `
  --kernel gbp_persistent `
  --incoming-layout exact `
  --schur-kernel inverse `
  --coop-block-policy work_cap `
  --persistent-threads 32 `
  --sweeps 100 `
  --fixed-lambda-start 60 `
  --hybrid-cycles 20 `
  --group-size 20 `
  --huber-delta 5
```

Wall time is sensitive to GPU power state, CHOLMOD DLL cold loading, CPU power
policy, and memory-transfer state. Run several fresh measurements and report
the distribution rather than treating `66.67 ms` as portable.
