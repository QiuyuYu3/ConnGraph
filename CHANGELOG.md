# Changelog

## [0.3.0] - 2026-10-09

### Changed

- **The package is renamed from `brainnet3d` to `conngraph`.** Use `import conngraph`, and `CONNGRAPH_DATA` in place of `BRAINNET3D_DATA`.
- Everything installs with the package; there are no optional extras.
- Saved figures are 300 dpi.
- Result tables, including those of `compute_graph_metrics(output_dir=...)`, are tab-separated `.tsv` files.
- Parallel work starts fresh worker processes on every platform, so a script that runs it needs an `if __name__ == "__main__":` guard on Linux too.

### Added

- A `conngraph` command in BIDS App style: the participant level computes graph metrics per participant, the group level collects them into tables and a report and compares groups. It reads XCP-D, NIRSPipe or MNE-Connectivity outputs, or a folder of matrix or time series files; `--modality` is required where the input does not say whether it is fMRI, fNIRS, EEG or MEG, and sets the report's wording.
- EEG, MEG and fNIRS input from MNE-Connectivity files: the method is read from the files, each method, frequency band and fNIRS chromophore is analysed separately, and phase and coherence measures are averaged and compared without Fisher z. Electrodes with standard names are drawn at the positions of an MNE template in MNI space.
- HTML reports for each participant, for graph metrics and for NBS, built from the result files they link to and rebuilt with `--reports-only`, with interactive figures, static figures saved beside the report (and rotatable 3-D brain pages with `--interactive-brain`), and a Methods section written from the run's settings.
- More input formats (unlabelled matrices, CIFTI, AFNI, nilearn), and connectivity computed from regional time series.
- Two more ways to build the graph (the disparity filter and PMFG), and more metrics, computed by default: nodal and global efficiency, path length, closeness and eigenvector centrality, participation coefficient, within-module degree z, modularity on the atlas networks or Louvain modules, and the small-world index.
- Hover cards, an edge threshold slider and network toggles in the interactive window; network-bundled circos plots.
- Brain figures can use any template: a pair of `.surf.gii` files or a brain volume (NIfTI or AFNI), whose outline is used.
- Two-group comparisons of graph metrics, network connectivity and edges with permutation t-tests, covariates, and FDR and family-wise corrected p-values (`compare_groups`), with an HTML report.
- NBS permutations run in parallel.
- A documentation site with a user guide, a CLI reference and a gallery of examples run on simulated data, and `datasets.make_mock_dataset` to simulate such data.

### Fixed

- Tube edges ignored the chosen colour, and multi-view colour bars had crowded labels.
- A node table without networks stopped the run; now only the node level is computed. Matrices that are not symmetric are refused instead of being read in part. TMFG on fewer than four nodes, as at a network level with two networks, now says so instead of failing with an index error.

## [0.2.0] - 2026-10-08

### Added

- Multi-view figures, node and edge highlighting, edge bundling and network layouts for the 3-D and spring plots.
- More ordering and styling options for heatmaps, circos and spring plots.
- Graph metrics: ten ways to build the graph, metric variants, integration over a range of densities or thresholds, signed weights and normalization against random networks; each run's options are saved with the results.

### Changed

- **Graph helper functions moved from `brainnet3d.viz` to `brainnet3d.graph_theory`.** Top-level imports are unchanged.
- **Metric tables have two-level columns (`result.node_df["strength.abs"]`) and are saved as one CSV per level and metric; network-level metrics use the hemisphere-split graph by default; NaN or Inf input raises an error.**
- Faster metrics, NBS and interactive window.
- Clearer category colours and network labels across plots.

### Fixed

- Layout, highlighting and display errors in the 3-D, spring and circos plots.

## [0.1.0] - 2026-10-07

### Added

- First release.
