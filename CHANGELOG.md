# Changelog

## [Unreleased]

### Added

- Edge colour range option and a ForceAtlas2 layout in the 3-D plot.
- A `verbose` option to silence progress messages.
- Hemisphere-split metrics work with any atlas that has a hemisphere column.

### Changed

- **Graph helper functions moved from `brainnet3d.viz` to `brainnet3d.graph_theory`.** Top-level imports are unchanged.
- **Graph metrics raise an error on NaN or Inf input** and report subjects that fail.
- **Metric tables have two-level columns, so `result.node_df["strength"]` gives one metric for all ROIs.** Saved CSV files are unchanged.
- Data problems are shown as warnings instead of printed text.

### Fixed

- Misplaced nodes with graph layouts, wrong edges highlighted when one hemisphere is shown, and display errors in spring and circos plots.

## [0.1.0] - 2026-10-07

### Added

- First release.
