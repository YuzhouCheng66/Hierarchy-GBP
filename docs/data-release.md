# Data release validation

The public release contains the nine canonical PGO graphs and ten main BAL
problems selected by `configs/datasets.json`. It does not contain the
supplementary Alamo input. `configs/data_release.json` pins the Hugging Face
commit and manifest checksum; source attribution and input preparation are
documented in the dataset card and notices.

## September 25 validation

- All 19 files passed structural checks before packaging. The individual gzip
  archives total 312,741,370 bytes and carry no original filename or timestamp.
- Local decompression and a second, unauthenticated download from the public
  repository both reproduced every canonical SHA-256 exactly.
- H-GBP FR079, Sphere and Ladybug49 passed the existing numerical reference checks
  at both 1 and 16 threads using those publicly downloaded inputs.
- All five PGO baseline adapters completed FR079 with archived binary/runtime
  fingerprints checked and independent full-information final-pose scoring.
  AMM and Cycle used their retained WSL binaries; the other three were native.
- All five BA baseline profiles completed Ladybug49 with archived executable
  hashes checked. Costs and reprojection metrics were read and checked from
  their native logs.
- The existing 19 compiled C++ tests passed. The 87 Python tests passed except
  one skipped Windows symlink-privilege test. Scoring tests used NumPy and SciPy.

These are data integrity and adapter smoke checks, not a new full-suite timing
study or a fresh build of every external baseline. Solver source files and
numerical policies were not changed. External dependency builds, objectives,
and timing boundaries remain explicitly documented under `baselines/`.

```powershell
python scripts/download_datasets.py --verify-only
python -m unittest discover -s tests -p "test_*.py"
ctest --test-dir build --output-on-failure
```
