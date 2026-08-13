# Reference results

Only publication-facing reference artifacts are tracked here. Raw benchmark
runs are machine-specific and are written below `results/reproduction` by
default.

- `pgo_comparison_with_hgbp1.tex`: pose-graph table source.
- `pgo_time_normalization_20260727.json`: PGO timing normalization provenance.
- `ba_comparison_with_hgbp1.tex`: bundle-adjustment table source.
- `ba_time_normalization_20260727.json`: BA timing normalization provenance.
- `validation_20260813.json`: clean-build PGO/BA accuracy and wall-time audit.

Every fresh run records the resolved parameters, input hash, command line,
wall time, and numerical checks in JSON. The formal reference values and
tolerances are defined in `configs/pgo.json` and `configs/ba.json`.
