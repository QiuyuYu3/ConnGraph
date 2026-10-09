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
brainnet3d-graph derivatives/xcpd results/graph --input-format xcpd --atlas Gordon

brainnet3d-nbs derivatives/xcpd results/nbs --input-format xcpd --atlas Gordon \
    --groups participants.tsv --group-column group --contrast PT HC --thresh 3.5 --seed 1
```

## Input

`--input-format` is required and says what `INPUT` holds:

| Format | `INPUT` |
|---|---|
| `xcpd` | an XCP-D derivatives folder; choose the atlas with `--atlas` |
| `matrix` | a folder of connectivity matrices, laid out as below |
| `timeseries` | a folder of regional time series, laid out as below; connectivity is computed first (`--connectivity`) |

A `matrix` or `timeseries` folder holds one file per participant and a node table with fixed names:

```
INPUT/
    nodes.tsv                  label, network; optionally hemisphere, x, y, z
    sub-01_matrix.<ext>        or sub-01_timeseries.<ext>
    sub-02_matrix.<ext>
    ...
```

- `<ext>` is `.tsv`, `.csv`, `.txt`, `.1D`, `.npy` or `.mat`, or a CIFTI `.pconn.nii` (matrices) or `.ptseries.nii` (time series).
- A matrix is square, one row and one column per node. A time series has one column per node and one row per time point.
- Tables may carry the node labels as headers; without them, rows and columns follow the order of `nodes.tsv`.
- Matrices hold Pearson correlations; add `--values z` if they hold Fisher z values.
- `nodes.tsv` has one row per node. Without x, y, z the report leaves out the brain figures. For XCP-D input with the Gordon atlas, coordinates are added automatically.

## Output

`brainnet3d-graph` writes one table per level and metric (`node/`, `network_hemi/`, ...) and `graph_report.html`. `brainnet3d-nbs` writes `nbs_components.tsv`, `nbs_edges.tsv`, `nbs_null.tsv` and `nbs_report.html`.
