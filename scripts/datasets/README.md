# Synthetic Globe SE3 pose graphs

This directory regenerates the exact synthetic `Globe10k` and `Globe100k`
pose-graph inputs referenced by `configs/datasets.json`. These graphs were
reconstructed for the H-GBP experiments; they are not official upstream
dataset files.

## Reference inputs

| Dataset | Grid | Nodes | Factors | Extra edges | Edge mode | SHA-256 |
|---|---:|---:|---:|---:|---|---|
| Globe10k | 100 x 100 | 10,000 | 20,899 | 999 | `medium_chords` | `3f0b6f2747a1cfbee82500f707c7000b8d404e2cd76e057f5184c94236749c52` |
| Globe100k | 316 x 316 | 99,856 | 200,395 | 999 | `local_diag` | `86045ecc9355c0bf91b87b1dd82fe6391a216bf6baf6d2123f6cf3dad1a56b8d` |

`Globe100k` has 99,856 rather than exactly 100,000 vertices because it uses
a 316 x 316 globe grid.

## Generate

Install the data-generation dependency from the repository root:

```powershell
python -m pip install -r .\scripts\datasets\requirements.txt
New-Item -ItemType Directory -Force .\data\pgo
```

Generate the formal benchmark inputs:

```powershell
python .\scripts\datasets\generate_globe_like_se3.py `
  --preset globe10k `
  --edge-mode medium_chords `
  --out-g2o .\data\pgo\Globe10k.g2o

python .\scripts\datasets\generate_globe_like_se3.py `
  --preset globe100k `
  --edge-mode local_diag `
  --out-g2o .\data\pgo\Globe100k.g2o
```

The omitted options are part of the formal definition: seed `7`, translation
and rotation information `400`, noise multiplier `1`, legacy grid-first edge
order, and smooth-distortion initial poses. Each command also writes a
sidecar `.meta.json` containing every resolved parameter.

Verify the generated files before benchmarking:

```powershell
Get-FileHash .\data\pgo\Globe10k.g2o -Algorithm SHA256
Get-FileHash .\data\pgo\Globe100k.g2o -Algorithm SHA256
```

Both hashes were reproduced byte-for-byte with Python 3.11 and NumPy 2.4.6.

## Optional odometry initialization

`reseed_globe_odometry.py` creates an alternative odometry-chain initial
trajectory while preserving every edge measurement and information matrix.
It also emits consecutive-ID odometry edges before loop closures:

```powershell
python .\scripts\datasets\reseed_globe_odometry.py `
  --input .\data\pgo\Globe10k.g2o `
  --output .\data\pgo\Globe10k_odometry.g2o
```

The reseeded file has a different SHA-256 and is not the formal benchmark
input listed above.
