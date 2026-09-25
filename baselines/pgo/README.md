# Measured PGO baseline replay

These are the September 23 retained baseline policies, not interchangeable stock
solver commands. This package does not download dependencies, run builds on import,
or claim new timing/quality results. Inputs come from the release's
`configs/datasets.json`: all nine canonical PGO hashes, including Cubicle PSDfloor1.
Use the PGO directory restored by the release downloader as `--data-root`.
The public data package is
[HGBP-Benchmarks](https://huggingface.co/datasets/yuzhoucheng66/HGBP-Benchmarks),
revision `e57f8d72b4dd8f02d9d17d578696aa552b4e10af`.

## Methods and limits

- `cholmod1`: custom matched analytic Lie-log frontend, official g2o CHOLMOD kernel,
  one thread, 20 full outers, Huber delta 5, hard first-pose gauge, jitter 1e-10,
  block ordering and symbolic reuse. **Not stock g2o CLI.** The measured archived
  CHOLMOD had supernodal support compiled out. This package does not deliberately
  disable a faster backend in a new dependency build. A modern rebuild must be
  reported as a rebuild, not the measured archived backend.
- `pcg16`: custom matched Lie-log vertices/edges, official OpenMP g2o GN frontend,
  **serial** fixed-block inner PCG. Frontend budget 16; 20 outers; Huber delta 5;
  historical relative tolerance 1e-6, absoluteTolerance=false, maxIter=-1 (scalar
  system dimension). No CHOLMOD-wrapper diagonal jitter. No claim of 16-way PCG.
- `amm16`: pinned DPGO optimizer plus the retained parallel-node/timing/CLI patch.
  Sixteen logical partitions and 16-worker budget. Its scalar-information chordal
  Huber objective (`loss_reg=25`) and initialization differ from the matched
  Lie-log problem. Original iteration budgets remain FR079=100, Grid=20,
  Globe100k=100, all others=300. Local iteration/accepted limits 10/1 and gradient
  tolerances 1e-3/1e-4 are unchanged. Do not use ideal `time / num_nodes` as wall time.
- `cycle16`: pinned upstream EC_PGO, MCB, maximum 20 iterations, threshold 0,
  16-worker budget, **nonrobust** native objective. No Huber is silently added.
  Shared reported costs used BFS reconstruction from optimized edges for every
  scene; official vertex reconstruction assumes consecutive odometry (not true
  for Cubicle). The retained conversion helper is included, not a new optimization.
- `schwarz16`: independent GDSW additive two-level Schwarz-PCG adaptation, not the
  authors' official implementation. Sixteen subdomains/workers, overlap 1,
  local shift/jitter 1e-10, PCG relative tolerance 1e-8 and dimension cap, original
  line search and gradient stopping 1e-8 absolute / 1e-6 relative. Early termination
  is preserved, not forced to 20. This also differs from the paper's residual model.

Cycle/Schwarz Globe100k were **not freshly measured** in the retained retest.
The runner emits `historically_unmeasured` records by default, never fabricated
times/failures. `--include-unmeasured` opts into genuinely new attempts using the
same policy. The historical Schwarz evidence is an 1800-second timeout, not a
fresh >2000-second observation. Cycle evidence includes SIGSEGV under a 24-GiB
address-space cap, not proof of unrestricted physical OOM.

## Run existing binaries

Use Python 3.10-3.12 from the release root. **NumPy and SciPy are required for all
real runs**, including Cycle: independent scoring is now mandatory after a
successful solve. A dry-run needs only the Python standard library. Install the
tested scoring dependencies in your chosen native/WSL environment:

```bash
python -m pip install -r baselines/pgo/requirements.txt
```

`--binary-root` contains a single method's
executables and compatible runtime, **not** the H-GBP build directory. Native
executable names are in `policies.json`; Windows adds `.exe`. Explicit paths take
precedence over `HGBP_PGO_DATA_ROOT`. Output directories must not already exist.

Small smoke commands (each uses the unchanged retained iteration budget):

```powershell
# Set these to your HF PGO data and separate retained/rebuilt runtime directories.
python baselines/pgo/run.py cholmod1 FR079 --data-root "$data" --binary-root "$cholmod" --output-root "$out/cholmod-fr079" --timeout 2000
python baselines/pgo/run.py pcg16 FR079 --data-root "$data" --binary-root "$pcg" --output-root "$out/pcg-fr079" --timeout 2000
python baselines/pgo/run.py schwarz16 FR079 --data-root "$data" --binary-root "$schwarz" --output-root "$out/schwarz-fr079" --timeout 2000
```

Append `--dry-run` to hash/validate and write commands without launching anything.
Use a different output directory for the subsequent real run. Dataset names may
also follow options. Omit names to select all nine; use `--repeats N` for sequential
repetition. There are deliberately no numerical-tuning or thread-count switches.

Run AMM/Cycle **inside Linux/WSL**, with Linux paths to the same canonical bytes:

```bash
python3 baselines/pgo/run.py amm16 FR079 --data-root "$data" --binary-root "$amm/bin" --library-root "$amm/lib" --output-root "$out/amm-fr079" --timeout 2000
python3 baselines/pgo/run.py cycle16 FR079 --data-root "$data" --binary-root "$cycle/bin" --library-root "$cycle/lib" --output-root "$out/cycle-fr079" --timeout 2000
```

Linux requires Bash, GNU time/timeout and taskset. The measured 24-GiB address-space
cap and default CPU mask 0xffff (CPUs 0-15) are preserved. `--affinity-mask MASK` or
`--no-affinity` is a recorded hardware change, not algorithm tuning. WSL vCPUs are
not asserted to be the same physical cores as Windows mask 0xffff. GNU timeout
terminates the solver process group. Native/launcher wall times remain separate.

The runner records commands, selected environment, input/executable/bundle hashes,
native counters/objectives, independently rescored costs, output hashes, failures
and timeouts. Numerical inherited
environment overrides are cleared. `--require-archived-runtime` requires the retained
bundle fingerprints; otherwise mismatches are explicitly **rebuilt/unverified**, not
quietly called archived runs. System/transitive libraries are not fully fingerprinted.
No timing ratio is a cross-machine acceptance test.

## Build small native wrappers

Only the small measured adapter/geometry sources are included. g2o, Eigen and
SuiteSparse remain external. `provenance.json` names snapshots, upstream pins and
retained runtime hashes. The g2o base tag is `20241228_git`; installed vcpkg patches,
Eigen ABI, OpenMP and CHOLMOD features also matter. A tag alone does not recreate
the original binary. Keep the native and OpenMP g2o installs distinct.

```powershell
cmake -S baselines/pgo -B "$build" -DCMAKE_BUILD_TYPE=Release -DPGO_G2O_ROOT="$g2oNative" -DPGO_EIGEN_ROOT="$eigenNative" -DPGO_G2O_OPENMP_ROOT="$g2oOpenMP" -DPGO_PCG_EIGEN_ROOT="$eigenOpenMP"
cmake --build "$build" --config Release --target cholmod_se2 cholmod_se3 g2o_pcg_openmp g2o_block_pcg_se2 g2o_block_pcg_se3 --parallel 2
```

Outputs/runtimes are separated into `cholmod1/`, `pcg16/`, `schwarz16/` beneath
the build directory. Both groups are optional: omit a g2o root to omit its targets.
The archived MSVC CHOLMOD/Schwarz AVX2 flag is selectable with
`-DPGO_ARCHIVED_AVX2=OFF` for other CPUs; report that hardware/build change.
This CMake file has not been compiled during packaging.

## External upstream builds

Use clean external checkouts at the exact commits in `provenance.json`. No absolute
developer paths are needed. For AMM, apply `patches/amm-driver.patch` with
`git -C "$ammSource" apply "$release/baselines/pgo/patches/amm-driver.patch"`, then:

```bash
cmake -S "$ammSource/C++" -B "$ammBuild" -DCMAKE_BUILD_TYPE=Release -DENABLE_FAST_INSTRUCTIONS=ON -DENABLE_OPENMP=ON
cmake --build "$ammBuild" --target dist_pgo --parallel 2
```

The patch exposes the same upstream parameters, parallelizes independent per-node
driver phases with the communication barrier retained, and adds honest native
wall timers. It is not an unchanged stock example. External dependencies include
Eigen, Boost program_options, glog, BLAS/LAPACK, SuiteSparse/CHOLMOD/SPQR.

For Cycle, use its pinned external source root and original build system. The
optional retained `patches/cycle-output-precision.patch` changes text export to
17 significant digits, not the optimizer. The protocol fingerprinted Main.cpp and
ECPGO_NonlinearSolver.h but **not this export header**. The complete historical
Cycle toolchain/build recipe was not archived here, so no unverified generic
build command is presented as an exact rebuild. Use the retained binary bundle
for exact measured replay; report newly built binaries as such.

## Scoring and validation

Successful runs include `independent_scores.json`: original full-information
Lie-log half-SSR and half-Huber(delta=5), excluding the separately reported soft
anchor diagnostic. The retained vectorized SE2/SE3 scoring and AMM conversion
functions are extracted with source hashes in `scoring_provenance.json`; no new
optimizer or objective is introduced. Native matched-frontend costs must agree
with independent costs within relative 1e-8. AMM/Cycle native objectives remain
separately labeled, never compared directly to H-GBP costs. Scorer hashes,
NumPy/SciPy versions, input/pose hashes, and scoring/conversion wall times are saved.

Cholesky/Schwarz
timing differs: Schwarz's reported `total_sec` includes parsing and all setup;
its separate `solve_sec` excludes all setup. AMM uses measured setup+optimization;
Cycle reports algorithm time and optimization+vertex time separately.

Cycle BFS conversion is automatic for every scene before scoring. Its standalone
interface is retained:

```bash
python3 baselines/pgo/reconstruct_cycle_vertices.py --edges "$out/cycle-fr079/FR079_r1/edges.txt" --dimension 2 --num-nodes 989 --output "$out/cycle-fr079/FR079_r1/reconstructed.g2o" --report "$out/cycle-fr079/FR079_r1/reconstruction.json"
```

The driver requires `connected_components == 1`. Conversion and complete rescoring
wall times are separate from native and process wall times. AMM exports
`estimates_huber.txt`: first N rows are translations; remaining rows contain the
transposed rotation matrices, shape `((dimension+1)*N, dimension)`. Shared scoring
uses original full-information factors, half-SSR and half-Huber(delta=5), with
the retained Lie-log convention. Existing outputs can be rescored without running
any solver (use a new score output filename):

```bash
python baselines/pgo/rescore.py cholmod1 FR079 --data-root "$data" --run-dir "$out/cholmod-fr079/FR079_r1" --output "$out/cholmod-fr079/independent-check.json"
python -m unittest discover -s tests -p test_pgo_baselines.py -v
```

Packaging validation: 30 mocked/known-answer tests passed, including full-information
SE3 cross terms, half-SSR/Huber normalization, AMM rotation layout, and Cycle BFS.
A read-only parser audit accepted 43 retained result sets (9 CHOLMOD, 9 PCG,
8 Schwarz, 9 AMM, 8 Cycle); it did not execute solvers. The release integrator
also reported successful strict-runtime, independently scored FR079 smoke runs
for all five methods: CHOLMOD, PCG, Schwarz, AMM and Cycle. Its newer isolated
environments used NumPy 2.4.6 / SciPy 1.17.1 for the native methods and SciPy
1.18.1 for WSL methods; the pinned requirements above are the environment used
for packaging's unit fixtures, not an algorithm restriction. These small
integration checks are not a new all-nine benchmark.

Remaining gaps: wrappers have not been freshly compiled during packaging;
the full historical compiler/system-library stack and Cycle build recipe are not
fully archived. No byte-identical source-rebuild or cross-machine speed claim is made.

No binaries, libraries, datasets, benchmark outputs, or huge dependency trees are
vendored. No broad numerical benchmark or heavy build was run during packaging.
See `NOTICE.md` and `licenses/` before distributing dependency binaries.
