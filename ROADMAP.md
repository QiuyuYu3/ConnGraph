# Roadmap

## Vision

Turn a connectivity matrix and a node table into graph-theory metrics and an interactive 3-D brain view, whatever pipeline produced the matrix.

## Completed Features

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

## Milestones

- **Faster NBS**: run permutations in parallel.
- **Hover popup**: show node label and metrics when hovering over a node.
- **Interactive filtering**: edge-threshold slider and per-network show/hide.
- **EEG and fNIRS input**: accept channel-, sensor- or source-level connectivity from EEG and fNIRS (design under discussion).
