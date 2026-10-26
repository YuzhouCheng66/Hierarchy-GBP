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
`tools/render_teaser.py` independently renders the README GIF from the same
surface model, with a white background and supersampled curves for GitHub.
Neither illustration represents empirical solver trajectories.

`data/` contains fixed, uniformly sampled real reconstruction and pose states,
with costs and clocks in `ba.json` and `pgo.json`. Geometry capture tools live in
`tools/`; optional native observers are built through `integrations/`. Captures
do not change solver updates. They add I/O and are not performance runs.
The BA clock uses archived native iteration-time fractions, scaled to the paper
totals. Its geometry and clocks come from separate verified executions. The PGO
demo uses fresh native clocks from continuous runs on Globe10k initialized by
integrating the odometry chain. It does not replace the paper's benchmark rows.

PowerBA rejected steps retain their preceding accepted state. g2o's first eight
GN states are checkpointed through 17-digit g2o files and checked against
the new continuous-run costs; its final state is from that continuous run.
Uncaptured g2o iterations are not presented as recorded geometry.

Display interpolation, a common nonlinear time axis, fixed initial-gauge
alignment and geometry sampling are presentation only. Cost readouts always
use completed iterations. `tools/import_tables.py` preserves the paper tables'
displayed precision from a pinned repository revision.

Inset plots use the same cost and time limits for both methods, with logarithmic
cost and the shared nonlinear display clock. Markers are completed outers.
`data/ba-view.json` defines one proper rigid display rotation for the tower and
square, shared by both methods and all frames; it does not modify solver states.
`data/gallery.json` registers four large BA scenes and three PGO scenes. Captures
are lazy-loaded by the dataset selector; switching disposes the previous replay.
The URL parameters `?ba=final4585&pgo=globe100k` select shareable initial scenes.
Per-scene provenance distinguishes new native clocks, archived clocks and the
paper-calibrated replays. These visualizations do not replace benchmark timings.
Run `node site/tools/test_replay.mjs` and `node site/tools/test_gallery.mjs` for
replay controls, final costs, finite rendering, dataset switching and cleanup.
