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
brainnet3d-graph derivatives/xcpd results/graph --input-type xcpd --atlases Gordon

brainnet3d-nbs derivatives/xcpd results/nbs --input-type xcpd --atlases Gordon \
    --groups participants.tsv --group-column group --contrast PT HC --thresh 3.5 --seed 1
```

## Input

`--input-type` is required and says what `INPUT` holds:

| Type | `INPUT` |
|---|---|
| `xcpd` | an XCP-D derivatives folder; name one or more atlases with `--atlases` |
| `fnirs-pipe` | a fnirs-pipe derivatives folder; HbO and HbR are analysed separately (`--chromophore` picks one) |
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

With several sessions, the session label follows the participant: `sub-01_ses-01_matrix.<ext>`.

- `<ext>` is `.tsv`, `.csv`, `.txt`, `.1D`, `.npy` or `.mat`, a CIFTI `.pconn.nii` (matrices) or `.ptseries.nii` (time series), or AFNI `3dNetCorr` output: `.netcc` (matrices) or `.netts` (time series, one region per row).
- A matrix is square, one row and one column per node. A time series has one column per node and one row per time point.
- Tables may carry the node labels as headers; without them, rows and columns follow the order of `nodes.tsv`.
- Matrices hold Pearson correlations; add `--values z` if they hold Fisher z values.
- `nodes.tsv` has one row per node. Without x, y, z the report leaves out the brain figures. For XCP-D input with the Gordon atlas, coordinates are added automatically.

`--participant-label` and `--session-id` select participants and sessions, as in XCP-D; for `xcpd` and `fnirs-pipe`, `--task-id` selects the task. Each session is analysed on its own; without `--session-id`, every session in `INPUT` is.

## Output

`brainnet3d-graph` writes one table per level and metric (`node/`, `network_hemi/`, ...) and `graph_report.html`. `brainnet3d-nbs` writes `nbs_components.tsv`, `nbs_edges.tsv`, `nbs_null.tsv` and `nbs_report.html`.

For `xcpd` these go into one folder per atlas (`OUTPUT/atlas-Gordon/`), and for `fnirs-pipe` into one folder per chromophore (`OUTPUT/chromo-hbo/`, `OUTPUT/chromo-hbr/`). fNIRS channels have no networks, so `brainnet3d-graph` computes the node level only.

When `INPUT` has sessions, each gets its own folder above these (`OUTPUT/ses-01/atlas-Gordon/`), even if there is only one. If one session or atlas fails, the others still run, and its folder holds `error.txt`.
