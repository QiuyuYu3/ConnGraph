# Roadmap

## Vision

Turn a connectivity matrix and a node table into graph-theory metrics and an interactive 3-D brain view, whatever pipeline produced the matrix.

## Completed Features

### 0.3.0

- A `conngraph` command in BIDS App style for XCP-D, NIRSPipe, MNE-Connectivity, matrix and time series input, with a participant level and a group level.
- EEG and MEG connectivity from MNE-Connectivity, analysed band by band with each measure handled by name.
- HTML reports for each participant, for the group's graph metrics, for group comparisons and for NBS, built from the result files they link to and rebuilt with `--reports-only`.
- Two-group comparisons of graph metrics, network connectivity and edges with permutation t-tests and covariates.
- Brain figures on any template, given as surfaces or a brain volume.

### 0.2.0

- Graph metrics built in ten ways, with several variants of each metric, four ways to treat negative weights, integration over a range of densities or thresholds, normalization by random networks, and a parameters file saved with every run.
- Multi-view figures of the 3-D plot with colour bars and legends.
- Node and edge highlighting, edge colouring by sign, edge bundling and a network layout in the 3-D plot.
- Matrix heatmaps, circos and spring plots ordered and coloured by network.
- Smooth rotation in the interactive window with thousands of edges.

### 0.1.0

- Load single-subject or group connectivity matrices with a node table, dropping nodes that are mostly missing.
- Interactive 3-D plot of nodes and edges on a brain surface, with size and colour driven by any node column or by edge weight.
- 3-D figures render off screen by default; open an interactive window or save a standalone interactive HTML page on request.
- Click a node to highlight its edges; show one hemisphere; save screenshots.
- Use computed node-level metrics as node size or colour in the 3-D plot.
- Gordon 333-parcel node table and fsLR 32k surfaces, downloaded from their official sources on first use and cached.
- Node-level, network-level and hemisphere-split graph-theory metrics on TMFG-filtered graphs.
- Loaders for XCP-D correlation matrices.
- Network-based statistic (NBS) group comparison, with significant edges highlighted in the 3-D plot.
- 2-D spring and circos plots, 3-D spring layout, threshold and density graph filtering.
