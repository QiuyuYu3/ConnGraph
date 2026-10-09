# Changelog

## [Unreleased]

### Added

- Two more ways to build the graph for graph metrics: the disparity filter, also over a range of significance levels, and the planar maximally filtered graph (PMFG).
- NBS results record their options, groups, package versions and random seed; a run without a seed can be repeated with the recorded one.
- HTML reports for graph metrics and NBS results, with interactive figures, static and rotatable group-mean brain views, circle plots of the group network and of the significant edges, a spring layout of the group network, and a Methods section written from the run's settings, including where the matrices came from and which nodes the loaders dropped. plotly, jinja2 and bibtexparser are now required.
- Command-line tools `brainnet3d-graph` and `brainnet3d-nbs` read XCP-D outputs or a folder of matrix or time series files and write the tables, the parameter record, a dataset description and the report in one step. The input format must be named, and a matrix or time series folder holds `nodes.tsv` and one `sub-<label>_matrix` or `sub-<label>_timeseries` file per participant.
- The matrix loaders also read unlabelled matrices (text, NumPy, MATLAB files or arrays, in node table order) and CIFTI parcellated connectivity files, and accept Fisher z matrices.
- Connectivity matrices can be computed from regional time series in the same file formats (Pearson or partial correlation, optionally with shrinkage), directly or while loading, including from XCP-D time series. nilearn is now required.
- Hovering over a node in the interactive window shows a card with its label, network, hemisphere and the values used for node size or colour.
- The interactive window has an edge threshold slider and a network list that hides or shows each network.
- Legend titles of multi-view figures can be changed.
- Circos plots can bundle edges through their networks and colour each edge by the networks at its two ends.

### Changed

- Circos plots produce two figures by default: the curved chords coloured by weight as before, and the bundled, network-coloured version.
- Saved figures are 300 dpi: 2-D plots, multi-view figures by default, and 3-D screenshots, which are now rendered at twice the window size.
- 3-D edges drawn as tubes are thicker, so they read as tubes rather than lines.
- Everything installs with the package: graph metrics, atlas and surface templates, and HTML export are no longer optional extras.
- NBS permutations run in several processes by default (`n_jobs`); a given seed gives the same result whatever the number of processes.

### Fixed

- 3-D edges drawn as tubes take the chosen edge colour; they were always drawn gold.
- Colour bars in multi-view figures show three round tick values instead of crowded labels.

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
