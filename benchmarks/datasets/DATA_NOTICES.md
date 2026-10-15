# Data sources and notices

This collection preserves source attribution and distinguishes original
benchmark data from locally generated or prepared inputs. No single new
license is imposed on the entire mixed-source collection.

## BAL

Source: https://grail.cs.washington.edu/projects/bal/

Authors: Sameer Agarwal, Noah Snavely, Steven M. Seitz and Richard Szeliski.
Citation: *Bundle Adjustment in the Large*, ECCV 2010.

The official page distributes numerical test problems but does not specify
an explicit standalone dataset license. We do not infer one from Ceres or
RootBA's software license. Consult the original authors for uses requiring
additional permission. The upstream archive URL for each BA item is:

`https://grail.cs.washington.edu/projects/bal/data/<family>/<filename>.bz2`

where `<family>/<filename>` is the `path` field of that item in the manifest.

## Standard pose graphs

Reference collection:
https://github.com/MurpheyLab/DPGO/tree/da0157f09ab23ad92a3361ac95336e416d8588df/dataset

Original names are `FR079.g2o`, `FRH.g2o`, `M3500.g2o`, `parking-garage.g2o`,
`sphere2500.g2o`, `cubicle.g2o` and `grid3D.g2o`. The collection's MIT notice
is reproduced below; this does not assert that every underlying historical
dataset was originally authored by that repository's copyright holder.
FR079/FRH are separately archived TORO conversions, and Cubicle has the
documented offline information-matrix floor. The remaining four files are
byte-identical to the pinned reference collection.

TORO attribution: Giorgio Grisetti, Cyrill Stachniss, Slawomir Grzonka and
Wolfram Burgard. Source and usage information:
https://openslam-org.github.io/toro.html

### Reference distribution notice

MIT License

Copyright (c) Meta Platforms, Inc. and affiliates.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Synthetic Globe inputs

Globe10k and Globe100k were generated for these experiments. They are not
official replicas of the original paper's Globe datasets. Generator sources,
seed, edge policy and initial-pose policy are included in the code release.
