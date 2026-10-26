# Project Page

Buildless HTML/CSS/ES modules, deployed to GitHub Pages by
`.github/workflows/pages.yml`. No solver source is compiled into the website.

```powershell
python site/tools/serve.py
python site/tools/validate.py
```

Open `http://127.0.0.1:8787/`. The preview helper sets explicit JavaScript MIME
types because Windows registry defaults may otherwise prevent ES modules loading.

`mechanism.js` illustrates frequency-selective error correction;
`tools/render_teaser.py` renders the README GIF from the same surface model.
Neither illustration represents empirical solver trajectories.

`data/` contains fixed, uniformly sampled real reconstruction and pose states,
with costs and clocks in `ba.json` and `pgo.json`. Geometry capture tools live in
`tools/`; optional native observers are built through `integrations/`. Captures
do not change solver updates. They add I/O and are not performance runs.
The published clock uses archived native iteration-time fractions, scaled to
the corresponding paper totals. Geometry and clocks need not come from the same
execution. This is stated in each replay's provenance.

PowerBA rejected steps retain their preceding accepted state. g2o's first four
GN states are checkpointed through 17-digit g2o files and checked against
archived continuous-run costs; its final state is from that continuous run.
Uncaptured g2o iterations are not presented as recorded geometry.

Display interpolation, a common nonlinear time axis, fixed initial-gauge
alignment and geometry sampling are presentation only. Cost readouts always
use completed iterations. `tools/import_tables.py` preserves the paper tables'
displayed precision from a pinned repository revision.
