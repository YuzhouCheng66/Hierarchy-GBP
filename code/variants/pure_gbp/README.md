# GBP at every hierarchy level

This optional **all-GBP H-GBP** variant retains abstraction, recovery and loopy
Gaussian message passing, and replaces the default sparse coarse solve with
another GBP hierarchy. There is **no coarse direct or Krylov solve, direct
fallback, or fine-level direct polish**. Small dense factorizations remain
inside local Gaussian messages, block whitening and local basis construction;
this is not a factorization-free algorithm.

The package shares the repository's validated SE2/SE3 fine kernels rather than
copying a second solver implementation. It is selected at **build time**;
normal builds keep the default coarse solver. This experiment covers PGO, not BA.

## Layout

- `src/solver.cpp`: coarse context, initialization and correction interface.
- `include/pure_gbp/gbp.h`: exact block-whitened equation and GBP messages.
- `synchronous.h`: double-buffered, destination-owned second-level executor.
- `ordered.h`, `serial.h`, `packed_eta.h`, `eta_program.h`: top-level loopy
  message scheduling and cached fixed-precision eta updates. The spanning
  forest orders updates only; non-tree edges are not removed.
- `hierarchy.h`, `projection.h`: connected aggregation, guarded local modes,
  projected factor budgets and residual restriction/recovery.
- `config.json`: one coarse policy shared across every input.
- `tests/numerics.cpp`: executor, hierarchy, zero-RHS and failure checks.

No experimental DAG executor, alternative basis strategy, Aitken/Anderson
branch, data capture, or direct-solve fallback is included.

## Policy

Fine-level settings are inherited unchanged from `../../../benchmarks/configs/pgo.json`:
SE2 uses outer20 / c3 / k100 / group20 / r4; SE3 uses outer20 / c5 / k50 /
group20 / r12. Both use Huber5 and the main solver's adaptive precision policy.

Each fine coarse-correction call performs up to three residual-correction
cycles. A cycle uses 40 second-level GBP sweeps, then restricts the residual
to a third level with up to 400 GBP sweeps and prolongs its correction.
Corrections use the existing capped energy step. Upper messages freeze after
three successive precision changes below `1e-12`, then use cached eta maps;
residual checks are every eight sweeps, with tolerance `1e-8`.

Third-level aggregates have target size 16. Each retains `min(6, block_dofs)`
local modes. The partial eigensolver has residual/lowest-mode checks and a
full local eigendecomposition fallback. This is a **basis-construction**
fallback, not a coarse linear-solve fallback.

The second level uses the requested worker count. The top level uses its
native serial executor at scalar dimension <=256, and a persistent parallel
executor otherwise. All choices are structural, not dataset-name dispatch.
Matrix values and projected budgets are rebuilt for each new linearization;
precision caches are reused only within the current matrix generation.

## Build and run

Use the same dependencies and PGO runtime as the main repository:

```powershell
.\code\variants\pure_gbp\build.ps1 -DependencyRoot C:\deps -PGORuntimeDir C:\deps\pgo-runtime
ctest --test-dir code/variants/pure_gbp/build --output-on-failure -j 1
python code/variants/pure_gbp/run.py M3500 Sphere Cubicle Grid --threads 16 --output-root code/variants/pure_gbp/results/example
```

Download the pinned inputs using `python benchmarks/download.py --suite pgo`.
Use `--data-root` for existing inputs. The driver checks their SHA-256 hashes,
clears inherited numerical environment overrides, enforces the shared fine
policy and checks the executable's reported coarse backend. It records native
optimization wall time, process wall time, cost, commands and binary hashes.
Output directories must be new. `--trace` adds layer diagnostics and should
not be mixed with uninstrumented timing comparisons.

For paired measurements, build the default solver separately:

```powershell
.\code\build.ps1 -PGOOnly -BuildDirectory build-main -DependencyRoot C:\deps -PGORuntimeDir C:\deps\pgo-runtime
python code/variants/pure_gbp/run.py M3500 Sphere Cubicle Grid --threads 16 --main-build build-main --repeats 3 --output-root code/variants/pure_gbp/results/paired
```

The default build switch is OFF. The experiment wrapper passes
`-DHGBP_PURE_GBP=ON -DHGBP_BUILD_BA=OFF` and stages runtime DLLs separately.
It can also be configured with `cmake -S code/variants/pure_gbp -B <build>`
using the repository's usual dependency variables and compiler environment.

## Paired validation

The release evidence reports actual same-machine comparisons to the current
default H-GBP solver, not timings scaled from historical tables. A ratio above
one means the all-GBP variant is slower. All runs use 16 threads and outer20;
the first eight rows are three-run medians, while Globe100k has one pair.

| Dataset | Default H-GBP [s] | All-GBP H-GBP [s] | All-GBP/default | Cost ratio |
| --- | ---: | ---: | ---: | ---: |
| FR079 | 0.113 | 0.200 | 1.772 | 1.000002 |
| FRH | 0.128 | 0.222 | 1.738 | 1.000002 |
| M3500 | 0.301 | 0.678 | 2.250 | 1.004510 |
| ParkingGarage | 1.680 | 4.222 | 2.514 | 1.002355 |
| Sphere | 1.252 | 2.082 | 1.664 | 1.000000 |
| Cubicle | 4.637 | 8.856 | 1.910 | 1.000557 |
| Grid | 3.798 | 5.500 | 1.448 | 1.000528 |
| Globe10k | 5.278 | 9.911 | 1.878 | 1.000014 |
| Globe100k | 41.957 | 94.665 | 2.256 | 1.000003 |

These measurements demonstrate a working all-GBP hierarchy, not a speed
improvement over the default coarse direct solve. No timing ratio is claimed
against other solvers here.
See `evidence/summary.json` for the
measured times, costs and run counts, and `evidence/runs.json` for commands,
input/binary hashes and effective configurations. Historical timings are not
portable performance guarantees. Archived commands and source hashes retain
the repository paths used when those measurements were collected.

To regenerate the compact evidence archive from completed run directories:

```powershell
python code/variants/pure_gbp/archive.py <paired-run-directory> [<additional-run-directory> ...]
```
