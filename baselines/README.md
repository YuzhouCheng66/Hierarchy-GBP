# Baseline reproduction

These adapters replay the comparison implementations using the same
hash-locked inputs as H-GBP. They are isolated from the H-GBP library and do
not modify its algorithm, configuration or build dependencies.

- [PGO](pgo/README.md): g2o-CHOLMOD, g2o-PCG, AMM-PGO, Cycle-PGO and Schwarz-PGO.
- [BA](ba/README.md): Ceres Sparse Schur, Ceres Iterative Schur, RootBA-QR and PowerBA.

Start by running `python scripts/download_datasets.py` from the code root.
Build each external implementation with its documented revision and use
the corresponding runner. External sources are obtained from their authors,
not silently replaced with a different solver. Dependencies and large
executables are not vendored in this repository.

## Comparison contract

Use identical initial states and data hashes. Do not regenerate a different
graph or substitute a similarly named BA problem. Retain each baseline's
documented stopping conditions, iteration budget, objective convention and
threading scope. In particular, a parallel frontend does not make a serial
inner solver parallel.

Keep native solver wall time separate from process wall time, and preserve
failure, timeout and out-of-memory outcomes. Do not scale historical timing
cells or count data downloads/compilation as optimization. Evaluate quality
alongside time; unlike residual definitions cannot be made identical merely
by using the same input file. The per-family instructions record these
implementation differences rather than claiming all methods solve an
identical numerical objective.
