# File interfaces

Use the maintained solver on your own inputs, with the same shared policies as
the benchmarks. These adapters do not modify the numerical kernels.

## BAL: Bundle Adjustment

Standard BAL text uses a 9-parameter camera (angle-axis, translation, focal
length, two radial distortion coefficients) and indexed 2D observations.

```powershell
python integrations/run.py bal scene.txt --threads 16 --output-dir results/my-ba
```

To export an optimized BAL model, build the separate `ba_file` adapter with
the **same matching dependencies** as the production BA solver:

```powershell
.\integrations\build.ps1 -DependencyRoot C:\deps `
  -RootBADir C:\deps\rootba -RootBALibrary C:\deps\rootba\rootba.lib `
  -RootBAPrefix C:\deps\rootba-runtime\Library
python integrations/run.py bal scene.txt --export --output-dir results/my-ba-export
```

The result includes `optimized.txt`, metrics and the exact command. Export uses
standard BAL coordinates, undoing the internal camera-axis convention; global
gauge/scale may differ because the solver normalizes the reconstruction.
The adapter is generated at build time from the maintained BA entry and fails
if its narrowly defined export hooks no longer match.

**Ceres:** BAL files can be exchanged with its Snavely-camera examples. This is
not a drop-in backend for arbitrary `ceres::Problem` camera models or constraints.

**COLMAP:** export a text reconstruction using COLMAP's `model_converter`, then:

```powershell
python integrations/colmap_to_bal.py sparse-text scene.txt
```

The converter supports `SIMPLE_PINHOLE`, `SIMPLE_RADIAL` and `RADIAL`, subtracts
the principal point and converts camera axes. Unsupported camera models fail
explicitly. Shared COLMAP intrinsics require explicit `--independent-intrinsics`
consent, since BAL optimizes one camera per image. Conversion is input-only;
automatic reinsertion into a COLMAP database is not provided.

## g2o: Pose Graphs

```powershell
python integrations/run.py g2o scene.g2o --space SE3 --threads 16 `
  --export --output-dir results/my-pgo
```

SE2 and `SE3:QUAT` are supported. `optimized.g2o` replaces vertex estimates
while retaining edge measurements, information matrices and original IDs.
Results can be read back by g2o or GTSAM's g2o file utilities. This is file
interchange, not a native GTSAM optimizer plugin. The solver uses its documented
anchoring and Huber policy; arbitrary user-specified priors/FIX constraints are
not a general factor-graph API. Input and output directories are never overwritten.

`--build-dir`, `--ba-adapter`, `--threads 1` and `--no-affinity` select the
executable/runtime or hardware, not dataset-specific numerical parameters.
The separate model export and optional scene capture introduce I/O overhead;
use the production benchmark runner for performance measurements.

Run adapter tests with `python -m unittest discover -s integrations/tests`.
