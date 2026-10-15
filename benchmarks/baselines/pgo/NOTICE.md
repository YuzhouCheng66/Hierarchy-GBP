# Source attribution

`src/` contains the retained local matched-frontend and independent Schwarz
reproduction code from `cpp/experiments/pgo_reviewer_baselines/g2o_pcg`, plus the
CHOLMOD and OpenMP-PCG adapters recovered from the measured source snapshots.
Original comments are preserved. Only line endings were normalized on extraction.
No separate third-party license header was present on those local adapters; this
notice does not assign them an invented upstream license. Their redistribution
must be covered by the release project's own licensing decision.

The external g2o dependency is by Rainer Kuemmerle and contributors:
https://github.com/RainerKuemmerle/g2o . Its main library uses BSD terms; individual
components and linked SuiteSparse/CHOLMOD features carry their own notices and
licenses. No g2o library or upstream source tree is redistributed here. Keep the
licenses of the actual external binaries you choose; the runner's hashes are
provenance, not a redistribution permission.

`patches/amm-driver.patch` modifies the upstream MurpheyLab/DPGO example and its
example CMake file at the pinned commit in `provenance.json`. The optimizer remains
external. The upstream MIT notice (Copyright Meta Platforms, Inc. and affiliates)
is preserved verbatim in `licenses/DPGO-MIT.txt`.

`patches/cycle-output-precision.patch` modifies the text exporter in Fang Bai's
Cycle-Based PGO, https://bitbucket.org/FangBai/cyclebasedpgo . Its upstream GPL v3
license text is preserved verbatim in `licenses/Cycle-GPL-3.0.txt`; the patch does
not relicense that project. Upstream bundled components keep their own notices.

`reconstruct_cycle_vertices.py` is the original local BFS integration helper from
the reviewer baseline scripts, referenced by the archived pose-conversion policy.
Only its callable entry point was separated from the CLI for in-process scoring.
It is not Cycle-PGO solver code and performs no nonlinear optimization.

`retained_se2.py`, `retained_se3.py`, and `retained_poses.py` contain extracted
local independent-scoring functions, with original file hashes and extraction
details in `scoring_provenance.json`. Numerical formulas are retained; report CLIs
and machine-specific imports are omitted. These local scripts carry the same
release-project licensing caveat as the local C++ adapters above.

Schwarz is an independently written GDSW-style adaptation motivated by
arXiv:2603.08975v2, not an official implementation or an assertion of identical
paper settings. Solver labels and objective differences are retained in policies.
