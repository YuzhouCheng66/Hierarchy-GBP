# Cleanup revalidation

Revalidated on 2026-10-15. This checks that the cleaned implementation retains
the accepted solver's numerical behavior and performance. It is **not** a fresh
comparison with competing solvers or evidence of a new algorithmic speedup.
The [machine-readable record](recertification.json) contains full precision,
sample ranges, cost-history differences, work counters and provenance hashes.

## Protocol

- Native Release executables, outer20, unchanged shared family configurations
  and hash-verified canonical inputs, including processed PGO inputs.
- CPU: three interleaved old/new pairs per dataset/thread case, run sequentially
  without concurrent builds, profilers or other benchmarks. Tables report
  medians. `new/old` means cleaned H-GBP / reference H-GBP, **not** H-GBP / Direct.
- Solver wall includes setup and numerical work, excluding input parsing and
  output. Process wall is a separate measurement. A median ratio above 1.15 was
  the performance-review threshold, not a numerical acceptance criterion.
- i9-13900HX (24 physical cores, 32 logical processors), affinity `0xffff`,
  budgets of 1 or 16 threads; RTX 4080 Laptop, CUDA 12.8.61, driver 581.42,
  MSVC 19.44.35217.0. Sixteen threads does not mean sixteen physical cores.
- Numerical reference: `3dd8f91b458e90cd65c1086e658137bfc08bb412`.
  No C++ solver implementation or numerical-policy changes were made during
  this certification. External baselines and Globe100k were not timed again.

## Bundle adjustment

All ten datasets were rerun at both thread budgets: **120 native runs**.
The reference was the accepted September 24 executable. Expanding the cleaned
BA implementation fragments reproduces the pre-cleanup source exactly after
the private include-path substitution; that source also matches the archived
release. All adjacent runtime DLL hashes match between reference and rebuild.

The missing RootBA static library was rebuilt from its original nine source
files. All 148 pinned RootBA source/build files match their source lock. Matching
development packages were recovered using the accepted runtime DLL hashes;
the original static library's bitwise identity is not claimed. Package versions,
hashes and the rebuilt library hash are in the JSON record.

| Dataset | New 1t [s] | new/old 1t | New 16t [s] | new/old 16t | Cost 1t | Cost 16t |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Ladybug49 | 0.168662 | 0.9588 | 0.067328 | 0.9884 | 26705.214411 | 26705.214411 |
| Venice89 | 2.597304 | 1.0794 | 0.556372 | 0.9991 | 576781.270131 | 576781.270131 |
| Final93 | 1.486321 | 0.9903 | 0.365556 | 1.0076 | 291092.148433 | 291092.148433 |
| Trafalgar257 | 1.375184 | 1.0308 | 0.433368 | 0.9965 | 197378.482286 | 197378.482286 |
| Dubrovnik356 | 6.180397 | 0.9783 | 1.522670 | 1.0043 | 1021101.790756 | 1021101.790713 |
| Final394 | 2.783311 | 0.9868 | 0.779139 | 0.9965 | 598296.093880 | 598296.093880 |
| Ladybug1723 | 4.110556 | 1.0013 | 1.190229 | 1.0014 | 759599.684751 | 759599.168872 |
| Venice1778 | 25.397956 | 1.0241 | 5.600285 | 1.0019 | 3363149.165014 | 3363149.170648 |
| Final3068 | 10.759407 | 0.9575 | 2.931950 | 1.0059 | 3240217.079015 | 3240217.246311 |
| Final4585 | 48.415601 | 1.0279 | 10.845760 | 0.9954 | 10331431.914015 | 10331431.914014 |

All single-thread 21-row cost histories are exactly unchanged. Across
16-thread pairs, the maximum relative cost-history difference is
`2.905412e-6`; the maximum relative final-cost difference is `4.994288e-7`.
Small parallel differences are also present among reference repeats. All work
counters match, and all public cost/MRE/RMSE quality checks pass. These are
native reported BA metrics; BA geometry was not exported for an independent
rescore. The largest median slowdown is 7.94%, within the review threshold.

## Pose graph optimization

Sampled seven datasets across thirteen dataset/thread cases: **78 native
runs**, comparing the immediate pre-cleanup executable snapshot with the
cleaned build. Globe10k was sampled at 16 threads only. All paired cost
histories, work counters and full pose files are identical; runtime DLL hashes
also match. Median timing ratios range from 0.9902 to 1.0285.

| Dataset | New 1t [s] | new/old 1t | New 16t [s] | new/old 16t | Final cost (16t) |
| --- | ---: | ---: | ---: | ---: | ---: |
| FR079 | 0.285746 | 1.0285 | 0.108561 | 1.0011 | 18.785887503 |
| FRH | 0.341690 | 1.0084 | 0.116816 | 1.0129 | 9.656841047e-5 |
| M3500 | 0.857887 | 0.9954 | 0.280957 | 0.9902 | 68.957546676 |
| Sphere | 4.890810 | 1.0172 | 1.191436 | 1.0064 | 675.700962926 |
| Cubicle | 21.312819 | 0.9955 | 4.309231 | 0.9952 | 1378.534328656 |
| Grid | 16.407796 | 1.0097 | 3.692950 | 1.0024 | 42179.978945068 |
| Globe10k | Not sampled | -- | 5.430637 | 1.0182 | 3039.927579193 |

This sampling does not certify fresh timings for ParkingGarage, Globe100k or
Globe10k at one thread.

## Retained fast paths

Source selection was checked together with native settings and work counters,
not inferred from filenames alone. These checks cover the selected production
path, not every historical optimization experiment.

| Family | Selected implementation and observed work |
| --- | --- |
| SE2 | Every outer reports `packed_residual/default`: contiguous message slots, hand-written 3x3 full-lambda kernel, cached fixed-eta maps. The 1t route accumulates serial belief-eta deltas; the 16t default implementation retains a persistent team across sweeps. FRH-1t executed 1100 full-lambda and 4900 eta-only sweeps, including 4860 serial-delta sweeps. M3500-16t executed 2036 full-lambda, 3964 eta-only sweeps and 60 coarse solves, with no Schur fallback. |
| SE3 | The benchmark entry selects the packed workspace and disables per-pass profiling. One thread uses serial small-block kernels; multiple threads use the persistent sweep executor. The hand-written 6x6 forward-Schur kernel, fixed-eta maps, implicit fine operator, cached coarse assembly and symbolic-pattern reuse remain selected. Basis backend is `exact_band_selected_else_original_partial_eigen`; Sphere-1t made 2500 banded calls with zero failures. Legacy `enable_singlecore_fastsync=false` does not disable this packed serial route. |
| BA | Native settings report GBP fine smoothing, additive hierarchy correction and FCG control under the shared policy. Landmark-major buffers, retained cross-block caching, implicit exact Schur products and cached eta maps remain active. Fixed-eta sweeps use the persistent pair-slot executor; the full-lambda stage is not claimed to have that same multi-sweep persistent executor. Venice89-16t executed 100 full-lambda sweeps, 207 eta-only sweeps and 69 linear cycles. |

Relevant implementation: [SE2 kernels](../src/detail/se2_internal_residual.cpp),
[SE3 kernels](../src/detail/se3_internal_residual.cpp),
[SE3 basis](../src/detail/se3_basis.inc),
[SE3 entry](../apps/se3_benchmark.cpp),
[BA landmark data flow](../src/detail/ba_landmarks.inc),
[BA messages](../src/detail/ba_messages.inc), and
[BA solve](../src/detail/ba_solve.inc).

## Native CUDA

The retained [SE2 CUDA variant](../variants/se2_cuda_hgbp/README.md) was rebuilt
and run on the actual GPU. It is a hybrid: fine GBP runs on CUDA, while basis
and coarse work remain on CPU. This does not certify BA or SE3 CUDA.

The final comparison uses the **production CPU executable**, explicitly
selected with `-ReferenceBinary`, rather than the CPU reference target from
the independent CUDA build. A preliminary three-pair screen with that separate
target was retained but is not used for the final speed ratios. Nine alternating
measured CPU/GPU pairs follow an excluded warm-up for each dataset. Shadow and
sanitizer runs are separate from timings.

| Dataset | CPU16 solver [ms] | Hybrid solver [ms] | Solver speedup | CPU process [ms] | Hybrid process [ms] | Process speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FR079 | 113.6673 | 67.3293 | 1.6882x | 246.2441 | 294.3977 | 0.8364x |
| FRH | 124.0494 | 72.3817 | 1.7138x | 294.5694 | 334.4398 | 0.8808x |
| M3500 | 271.0629 | 142.5273 | 1.9018x | 579.1188 | 534.2482 | 1.0840x |

Every outer's CPU/GPU cost, pose, policy and work counters match exactly,
including comparisons with the old CUDA checkpoint. All 60 batch shadows per
dataset pass without CPU fallback. Memcheck passes on all three datasets;
synccheck passes on M3500, each reporting zero errors. Four native CUDA CTests
also pass, covering layout, coarse caching, overlap and failure recovery.
The lossless evidence archive verifies 33 paired trajectories.

**The 2x complete-process target is not met.** FR079 and FRH are slower when
launch, CUDA context, input/output and teardown are included. Context creation
measured about 72-81 ms and is already included in process wall; do not add it
again. Solver-only gains must not be presented as complete-process speedups.

## Regression and evidence

- Main compiled tests: 22/22 pass.
- Native CUDA compiled tests: 4/4 pass.
- Main Python tooling tests: 111 run, 110 pass, one Windows symlink-privilege skip.
- CUDA Python tests: 15/15 pass.

Local raw BA/PGO evidence is retained under
`E:/HGBP_profiles/recert_20261015/{ba,pgo}`; CUDA evidence and orchestration
logs are under the ignored `results/recert_20261015/` directory. Binary, input,
runtime, raw-record and archive hashes are recorded in the JSON. Those local
paths are provenance, not dependencies for using the repository. No competing
baseline results were rescaled or overwritten by this revalidation.
