# brainnet3d

Graph-theory metrics and network-based statistics for brain connectivity, with HTML reports and 3-D brain figures.

This package is in early development. The options may change, and documentation is in preparation.

## Installation

Clone this repository, then from its root:

```bash
pip install .
```

This installs two commands, `brainnet3d-graph` and `brainnet3d-nbs`.

## Commands

| Command | What it does |
|---|---|
| `brainnet3d-graph INPUT OUTPUT` | Graph-theory metrics for every participant, at the node level and the network level |
| `brainnet3d-nbs INPUT OUTPUT --groups TABLE --group-column COL --contrast A B --thresh T` | Compares two groups with the network-based statistic |

Each command writes its result tables, a `parameters.json` with every setting, a `dataset_description.json` and an HTML report. Run either command with `--help` for all options, such as the graph construction method, the metrics, or the number of permutations.

Examples on XCP-D output with the Gordon atlas:

```bash
brainnet3d-graph derivatives/xcpd results/graph --atlas Gordon

brainnet3d-nbs derivatives/xcpd results/nbs --atlas Gordon \
    --groups participants.tsv --group-column group --contrast PT HC --thresh 3.5 --seed 1
```

## Input

`--input-format` says what `INPUT` holds:

| Format | `INPUT` |
|---|---|
| `xcpd` (default) | an XCP-D derivatives folder; choose the atlas with `--atlas` |
| `matrix` | a folder of connectivity matrices, laid out as below |
| `timeseries` | a folder of regional time series, laid out as below; connectivity is computed first (`--connectivity`) |

A `matrix` or `timeseries` folder holds one file per participant and a node table, all tab-separated:

```
INPUT/
    nodes.tsv              label, network; optionally hemisphere, x, y, z
    sub-01_matrix.tsv      or sub-01_timeseries.tsv
    sub-02_matrix.tsv
    ...
```

- `sub-<label>_matrix.tsv`: a square matrix of Pearson correlations, with the node labels as the first row and the first column.
- `sub-<label>_timeseries.tsv`: one column per node, headed by its label, and one row per time point.
- `nodes.tsv`: one row per node. Without x, y, z the report leaves out the brain figures. For XCP-D input with the Gordon atlas, coordinates are added automatically.

## Output

`brainnet3d-graph` writes one table per level and metric (`node/`, `network_hemi/`, ...) and `graph_report.html`. `brainnet3d-nbs` writes `nbs_components.tsv`, `nbs_edges.tsv`, `nbs_null.tsv` and `nbs_report.html`.
