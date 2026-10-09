# ConnGraph

Graph-theory metrics and network-based statistics for brain connectivity, with HTML reports and 3-D brain figures.

This package is in early development. The options may change, and documentation is in preparation.

## Installation

Clone this repository, then from its root:

```bash
pip install .
```

This installs the `conngraph` command.

## Usage

`conngraph` works like a BIDS App, in two steps:

```
conngraph INPUT OUTPUT participant [options]
conngraph INPUT OUTPUT group [options]
```

| Level | What it does |
|---|---|
| `participant` | Computes graph-theory metrics, at the node level and the network level, and writes one set of files per participant. `--participant-label` limits it to some participants, so a cluster can run one job per participant. |
| `group` | Collects the participant results into tables and an HTML report. With a groups table, it also compares two groups with permutation t-tests, adjusting for `--covariates` if given: the graph metrics and the connectivity within and between networks by default, and every edge when `--compare` names `edges`. `--nbs-thresh` adds the network-based statistic. |

Graph options (method, metrics, random networks) belong to the participant level; report and group comparison options belong to the group level. Input options are given at both. Run `conngraph --help` for all options.

Example on XCP-D output with the Gordon atlas:

```bash
conngraph derivatives/xcpd results participant --input-type xcpd --atlases Gordon

conngraph derivatives/xcpd results group --input-type xcpd --atlases Gordon \
    --groups participants.tsv --group-column group --contrast PT HC --covariates age sex \
    --nbs-thresh 3.5 --random-seed 1
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

For files that differ in other BIDS entities, such as several runs or acquisitions, `--bids-filter-file` takes the same JSON file as XCP-D. Its `"bold"` entry (`"nirs"` for fnirs-pipe) names the entities the files must carry; a list accepts any of its values and `null` requires the entity to be absent. Task, space or session given there replace `--task-id`, `--space` and `--session-id`.

```json
{"bold": {"acquisition": "multiband", "run": 1}}
```

A participant with several runs left after filtering has each run analysed on its own (`01_run-1`, `01_run-2`), with a warning in the report; group comparisons stop instead, since they need one matrix per participant. XCP-D's `--combine-runs` merges runs before conngraph sees the data; for XCP-D output without it, `--combine-runs` together with `--connectivity` z-scores each run's time series and concatenates them in run order before computing connectivity.

## Output

The participant level writes, for each participant, session and atlas or chromophore, one table per level with a row per node or network and a column per metric, the network-level connectivity, and a `_metrics.json` with every setting:

```
OUTPUT/
    dataset_description.json
    sub-01/ses-01/
        sub-01_ses-01_atlas-Gordon_level-node_metrics.tsv
        sub-01_ses-01_atlas-Gordon_level-networkhemi_metrics.tsv
        sub-01_ses-01_atlas-Gordon_level-networkhemi_connectivity.tsv
        sub-01_ses-01_atlas-Gordon_metrics.json
    group/ses-01/atlas-Gordon/
        node/, network_hemi/, ...     one table per metric, a row per participant
        parameters.json
        graph_report.html
        compare/                      metrics_node.tsv, global_node.tsv, blocks_networkhemi.tsv, edges.tsv, ...
        nbs/                          nbs_components.tsv, nbs_edges.tsv, nbs_null.tsv, nbs_report.html
```

Without sessions the `ses-` folders are left out; matrix and time series input has no atlas folder. fNIRS channels have no networks, so only the node level is computed.

Each `compare/` table has one row per test with the t statistic (positive when the first group of `--contrast` is higher), its p-value, FDR-corrected and family-wise (max-T, `--n-perms` permutations) p-values, and the group means and sizes. Corrections apply within a family: one metric at one level, the whole-graph metrics of one level, the network blocks of one level, or all edges. Values missing or constant across participants are not tested.

Nodes with too many missing values are dropped by looking at every participant in `INPUT`, so participants run one at a time get the same nodes as a single run. The group level checks that all participants were run with the same options. If one session or atlas fails, the others still run, and `OUTPUT/logs/` (participant) or the group folder holds `error.txt`.
