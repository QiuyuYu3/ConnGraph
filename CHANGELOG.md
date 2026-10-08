# Changelog

## [Unreleased]

### Added

- Two more ways to build the graph for graph metrics: the disparity filter, also over a range of significance levels, and the planar maximally filtered graph (PMFG).
- NBS results record their options, groups, package versions and random seed; a run without a seed can be repeated with the recorded one.
- Hovering over a node in the interactive window shows a card with its label, network, hemisphere and the values used for node size or colour.
- The interactive window has an edge threshold slider and a network list that hides or shows each network.

### Fixed

- 3-D edges drawn as tubes take the chosen edge colour; they were always drawn gold.

## [0.2.0] - 2026-10-08

### Added

- Multi-view figures of the 3-D plot with colour bars and legends, at a chosen figure width (`plot_views`).
- Highlight chosen nodes or the nodes of highlighted edges, colour edges by sign, and bundle edges that run close together, in the 3-D plot.
- Matrix heatmaps and circos plots can order nodes by clustering, by a chosen network order or by how strongly networks connect; heatmaps can also draw boxes around networks.
- Spring plots can shade each network and colour edges by sign or weight.
- A network layout for the 3-D plot and the 2-D and 3-D spring plots that gives each network its own region.
- Edge colour range option in the 3-D, circos and spring plots, and a ForceAtlas2 layout in the 3-D plot.
- A `verbose` option to silence progress messages.
- Hemisphere-split metrics work with any atlas that has a hemisphere column.
- Graph metrics can build the graph in ten ways (TMFG stays the default), compute several variants of each metric, integrate metrics over a range of densities or thresholds, treat negative weights in four ways, scale each subject's weights by its largest weight, and divide each metric by its average over random networks; the options, package versions and random seed of each run are saved with the results.

### Changed

- **Graph helper functions moved from `brainnet3d.viz` to `brainnet3d.graph_theory`.** Top-level imports are unchanged.
- **Graph metrics raise an error on NaN or Inf input** and report subjects that fail.
- **Metric tables have two-level columns, and metric names include the variant, so `result.node_df["strength.abs"]` gives one metric for all ROIs.** Results are saved as one CSV file per level and metric.
- **Network-level metrics are computed on the hemisphere-split graph by default;** the whole-network graph is optional.
- The default clustering coefficient, betweenness and NBS permutations are computed much faster, with the same results.
- The interactive window rotates smoothly with thousands of edges; saved images and HTML pages are unchanged.
- Data problems are shown as warnings instead of printed text.
- Category colours no longer repeat when there are more than 12 categories, neighbouring categories get different hues, and each category keeps its colour across plots whatever the node order; numeric node colours default to viridis, and 3-D spring plots colour nodes by network when network labels are given.
- Node tables read from file keep the label "None" as text instead of treating it as missing, and network-level graph metrics leave out ROIs labelled "None" by default however the atlas was read.
- Matrix heatmaps and NBS plots show network colour strips, label networks instead of every ROI in large matrices, and leave a constant diagonal blank; the third NBS panel shows the group difference with significant edges marked instead of hiding the rest.
- Circos plots draw curved chords coloured and sized by weight, with an edge colour bar and a ring of network names in place of the legend, and keep the node table's order within each network.

### Fixed

- Misplaced nodes with graph layouts, wrong edges highlighted and the other surface still drawn when one hemisphere is shown, hatching on translucent brain surfaces, and display errors in spring and circos plots.
- Showing one hemisphere now raises an error when the node table has no hemisphere column, instead of drawing every node; networks named with numbers are ordered by value (2 before 10).

## [0.1.0] - 2026-10-07

### Added

- First release.
