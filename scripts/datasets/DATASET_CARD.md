---
pretty_name: H-GBP PGO and Bundle Adjustment Benchmarks
language:
- en
tags:
- robotics
- pose-graph-optimization
- bundle-adjustment
- numerical-optimization
size_categories:
- n<1K
configs:
- config_name: catalog
  data_files:
  - split: test
    path: metadata.csv
---

# H-GBP Benchmark Inputs

Exact numerical inputs for reproducing H-GBP and baseline experiments:
**9 pose graphs** and **10 bundle-adjustment problems**. Each compressed file
expands to the SHA-256-locked input used by the solver configurations.

## Download and run

Use `scripts/download_datasets.py` from the accompanying code release. It
downloads only the requested problems, verifies compressed and uncompressed
hashes, and restores `data/pgo` and `data/ba` without changing any values.

Alternatively download a `.gz` file and decompress it with a standard gzip
utility. `manifest.json` specifies its target path, SHA-256, dimensions and
preparation history. The Dataset Viewer shows the 19-problem catalog, not the
individual observations or graph vertices.

## Contents

- SE2: FR079, FRH, M3500.
- SE3: ParkingGarage, Sphere, Cubicle, Grid, Globe10k, Globe100k.
- BA: Ladybug49, Venice89, Final93, Trafalgar257, Dubrovnik356, Final394,
  Ladybug1723, Venice1778, Final3068, Final4585.

The two Globe graphs are project-generated synthetic inputs, not copies of
the similarly named datasets from another publication. Their exact generator
and configuration are distributed with the code. Globe100k has 99,856 nodes.

## Input definitions

All compared methods must use the same decompressed file, including its
initial estimates and information matrices. Do not substitute a different
upstream file with the same short name.

FR079 and FRH are the archived TORO-to-g2o conversions with the g2o information
ordering. Cubicle includes the fixed offline eigenvalue floor of 1 on its
6x6 information matrices. The other named non-Globe pose graphs have no
additional preparation recorded in the canonical manifest. Globe initial
poses use the generator's documented smooth-distortion initialization.

The BA inputs are the official BAL `pre.txt` files, byte-for-byte unchanged.
Landmark elimination, problem normalization and robust losses are solver
operations, not edits to these files. A Huber loss is not baked into the pose
graphs; reproduce the specified loss separately in each supported solver.

## Sources and attribution

BAL: Sameer Agarwal, Noah Snavely, Steven M. Seitz and Richard Szeliski,
*Bundle Adjustment in the Large*, ECCV 2010.
[Official data and camera convention](https://grail.cs.washington.edu/projects/bal/).

The standard pose graphs circulate in the g2o, TORO, SE-Sync and DPGO benchmark
collections. A pinned public reference distribution is
[MurpheyLab/DPGO](https://github.com/MurpheyLab/DPGO/tree/da0157f09ab23ad92a3361ac95336e416d8588df/dataset).
The packaged FR079/FRH precision and Cubicle preparation differ from that
distribution; the manifest, not the short filename, identifies this release.

See `NOTICE.md` for upstream notices. This mirror does not claim authorship
of the third-party datasets or grant a replacement license for them. There
are no source photographs, account identifiers or local filesystem paths in
the numerical files. These are benchmark inputs, not a train/test split or
evidence of held-out generalization.
