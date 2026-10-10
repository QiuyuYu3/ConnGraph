# User guide

## Two levels

`conngraph` works like a BIDS App, in two steps:

```text
conngraph INPUT OUTPUT participant [options]
conngraph INPUT OUTPUT group [options]
```

| Level | What it does |
|---|---|
| `participant` | Computes graph-theory metrics, at the node level and the network level, and writes one set of files and an HTML report (`OUTPUT/sub-<label>.html`, skipped with `--no-report`) per participant. `--participant-label` limits it to some participants, so a cluster can run one job per participant. |
| `group` | Collects the participant results into tables and an HTML report. With a groups table, it also compares two groups with permutation t-tests, adjusting for `--covariates` if given: the graph metrics and the connectivity within and between networks by default, and every edge when `--compare` names `edges`. `--correction` (`fdr`, `fwe` or `none`) is required for these comparisons. `--nbs-thresh` adds the network-based statistic. |

Graph options (method, metrics, random networks) belong to the participant level, where `--graph-method` is required; group comparison options belong to the group level. Input options and the report options `--coords`, `--surfaces`, `--interactive-brain` and `--no-report` are given at both. The [CLI reference](cli.md) lists the options of each level.

## Input

`--input-type` is required and says what `INPUT` holds:

| Type | `INPUT` |
|---|---|
| `xcpd` | an XCP-D derivatives folder; name one or more atlases with `--atlases` |
| `nirspipe` | a NIRSPipe derivatives folder; HbO and HbR are analysed separately (`--chromophore` picks one) |
| `matrix` | a folder of connectivity matrices, laid out as below |
| `timeseries` | a folder of regional time series, laid out as below; connectivity is computed first (`--connectivity`) |
| `mne` | a folder of EEG or MEG connectivity files saved by MNE-Connectivity; each frequency band is analysed separately (see [EEG and MEG](#eeg-and-meg)) |

### Matrix and time series folders

A `matrix` or `timeseries` folder holds one file per participant and a node table with fixed names:

```text
INPUT/
    nodes.tsv                  label; optionally network, hemisphere, x, y, z
    sub-01_matrix.<ext>        or sub-01_timeseries.<ext>
    sub-02_matrix.<ext>
    ...
```

With several sessions, the session label follows the participant: `sub-01_ses-01_matrix.<ext>`.

- `<ext>` is `.tsv`, `.csv`, `.txt`, `.1D`, `.npy` or `.mat`, a CIFTI `.pconn.nii` (matrices) or `.ptseries.nii` (time series), or AFNI `3dNetCorr` output: `.netcc` (matrices) or `.netts` (time series, one region per row).
- A matrix is square and symmetric, one row and one column per node; directed connectivity is not supported. A time series has one column per node and one row per time point.
- Tables may carry the node labels as headers; without them, rows and columns follow the order of `nodes.tsv`.
- Matrices hold Pearson correlations; add `--values z` if they hold Fisher z values, or `--no-fisher-z` if they hold a measure that is not a correlation.

### The node table

`nodes.tsv` has one row per node. Without a network column only the node level is computed; without x, y, z the report leaves out the brain figures. For XCP-D input with the Gordon atlas, coordinates are added automatically.

## Participants, sessions and runs

`--participant-label` and `--session-id` select participants and sessions, as in XCP-D; for `xcpd` and `nirspipe`, `--task-id` selects the task. Each session is analysed on its own; without `--session-id`, every session in `INPUT` is.

### BIDS filter files

For files that differ in other BIDS entities, such as several runs or acquisitions, `--bids-filter-file` takes the same JSON file as XCP-D. Its `"bold"` entry (`"nirs"` for nirspipe) names the entities the files must carry; a list accepts any of its values and `null` requires the entity to be absent. Task, space or session given there replace `--task-id`, `--space` and `--session-id`.

```json
{"bold": {"acquisition": "multiband", "run": 1}}
```

### Several runs

A participant with several runs left after filtering has each run analysed on its own (`01_run-1`, `01_run-2`), with a warning in the report; group comparisons stop instead, since they need one matrix per participant. XCP-D's `--combine-runs` merges runs before conngraph sees the data; for XCP-D output without it, `--combine-runs` together with `--connectivity` z-scores each run's time series and concatenates them in run order before computing connectivity.

## EEG and MEG

`mne` input reads connectivity computed with [MNE-Connectivity](https://mne.tools/mne-connectivity/) and saved with `conn.save()`, one file per participant:

```text
INPUT/
    nodes.tsv                  optional: label; network, hemisphere, x, y, z
    sub-01_connectivity.nc
    sub-02_connectivity.nc
    ...
```

```python
from mne_connectivity import spectral_connectivity_epochs

conn = spectral_connectivity_epochs(epochs, method="wpli", fmin=(4, 8, 13), fmax=(8, 13, 30), faverage=True)
conn.save("INPUT/sub-01_connectivity.nc")
```

### Methods and bands

- The method is read from the files and named in the report. Undirected methods are supported: `coh`, `imcoh` (taken as its absolute value), `plv`, `ciplv`, `ppc`, `pli`, `pli2_unbiased`, `wpli`, `wpli2_debiased` and envelope correlation. Directed methods such as `dpli`, `psi` or Granger causality are refused.
- Envelope correlation is Fisher z-transformed when averaged and compared, like other correlations; the phase and coherence measures are used as they are.
- Each frequency band averaged with `faverage=True` gets its own result folder, such as `band-8to13Hz`; files with single frequencies are refused. Envelope correlation has no band axis, so band-pass the data before computing it, and average its per-epoch matrices with `conn.combine()` before saving.

### Nodes

- Node names come from the files. `nodes.tsv` adds networks (lobes or regions, for example), hemispheres and coordinates; without a network column only the node level is computed.
- Brain figures need x, y, z in the template's space. Regions of a source-space atlas have them; scalp electrode positions lie outside the brain surface.

:::{important}
The phase-locking value and coherence are biased upward when there are few epochs. If epoch counts differ between groups, prefer a debiased measure such as `wpli2_debiased` or `ppc`; the report gives the range of epoch counts.
:::

Matrices of these measures computed elsewhere can be given as `matrix` input with `--no-fisher-z`.

## Output files

The participant level writes, for each participant, session and atlas or chromophore, one table per level with a row per node or network and a column per metric, the network-level connectivity, and a `_metrics.json` with every setting, plus one report per participant. The group level writes its tables, reports and figures under `group/`:

```text
OUTPUT/
    dataset_description.json
    sub-01.html                       the participant's report, a section per session and atlas
    sub-01/figures/                   its static brain views (300 dpi PNG)
    sub-01/ses-01/
        sub-01_ses-01_atlas-Gordon_level-node_metrics.tsv
        sub-01_ses-01_atlas-Gordon_level-networkhemi_metrics.tsv
        sub-01_ses-01_atlas-Gordon_level-networkhemi_connectivity.tsv
        sub-01_ses-01_atlas-Gordon_metrics.json
    group/ses-01/atlas-Gordon/
        node/, network_hemi/, ...     one table per metric, a row per participant
        correlation/                  network connectivity per participant, and node_mean.tsv, the group mean matrix
        nodes.tsv                     the node table the reports use, with coordinates when given
        parameters.json
        graph_report.html
        figures/                      the report's static figures (300 dpi PNG); keep them beside the report
        compare/                      metrics_node.tsv, global_node.tsv, blocks_networkhemi.tsv, edges.tsv, ..., compare_report.html
        nbs/                          nbs_components.tsv, nbs_edges.tsv, nbs_null.tsv, nbs_mean_group1.tsv, nbs_mean_group2.tsv, nbs_report.html, figures/
```

Without sessions the `ses-` folders are left out; matrix and time series input has no atlas folder. fNIRS channels have no networks, so only the node level is computed.

### Comparison tables

Each `compare/` table has one row per test with the t statistic (positive when the first group of `--contrast` is higher), its p-value, FDR-corrected and family-wise (max-T, `--n-perms` permutations) p-values, the group means and sizes, and whether the result is significant under the `--correction` you chose (FDR, family-wise or uncorrected; required, with `--alpha`, default 0.05).

Corrections apply within a family: one metric at one level, the whole-graph metrics of one level, the network blocks of one level, or all edges. Values missing or constant across participants are not tested.

### Missing nodes, option checks and failures

Nodes with too many missing values are dropped by looking at every participant in `INPUT`, so participants run one at a time get the same nodes as a single run. The group level checks that all participants were run with the same options. If one session or atlas fails, the others still run, and `OUTPUT/logs/` (participant) or the group folder holds `error.txt`.

## Reports

The reports are built from the files written to `OUTPUT`, so they always show what is on disk, and each figure and table links the files it comes from. `--reports-only` rebuilds them from those files without recomputing anything, for example after changing `--surfaces`; give the same input options as the original run, and no analysis options, which are read from the saved results.

### Figures

The reports link their static figures from the `figures/` folders, so keep those next to the reports when moving them. Brain figures are static images; `--interactive-brain` also saves each one as a rotatable 3-D page in `figures/`, linked below the image. The interactive figures and these pages load plotly from its CDN, so viewing them needs an internet connection; each has a camera button that saves it as PNG.

For publication figures at 300 dpi, use the plotting functions of the Python package (`BrainNetPlotter.plot_views`, `circos_plot`, `spring_plot`, `plot_nbs_matrices`), which save to any path at 300 dpi. The [gallery](auto_examples/index.rst) shows them.

### The brain template

Brain figures are drawn in the fsLR 32k surfaces. `--surfaces` takes another left and right `.surf.gii`, or one skull-stripped brain volume (NIfTI or AFNI BRIK/HEAD), such as a pediatric template, whose smoothed outline is used instead. Node coordinates must be in the template's space.

### Building many reports

Participant reports are built several at a time, up to `--n-jobs`. Each one needs about 1 GB of memory, so fewer run at once when they would not fit in `--mem` (in MB; default 90% of the machine's memory). If one participant's report fails, the others are still written, the error is saved as `report-sub-<label>.err` under `OUTPUT/logs/`, and the command exits with an error.
