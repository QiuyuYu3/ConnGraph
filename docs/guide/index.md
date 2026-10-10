# User guide

`conngraph` works like a BIDS App, in two steps:

```text
conngraph INPUT OUTPUT participant [options]
conngraph INPUT OUTPUT group [options]
```

| Level | What it does |
|---|---|
| `participant` | Computes graph-theory metrics, at the node level and the network level, and writes one set of files and an HTML report (`OUTPUT/sub-<label>.html`, skipped with `--no-report`) per participant. `--participant-label` limits it to some participants, so a cluster can run one job per participant. |
| `group` | Collects the participant results into tables and an HTML report. With a groups table, it also compares two groups with permutation t-tests, adjusting for `--covariates` if given: the graph metrics and the connectivity within and between networks by default, and every edge when `--compare` names `edges`. `--correction` (`fdr`, `fwe` or `none`) is required for these comparisons. `--nbs-thresh` adds the network-based statistic. |

Graph options (method, metrics, random networks) belong to the participant level, where `--graph-method` is required; group comparison options belong to the group level. Input options and the report options `--coords`, `--surfaces`, `--interactive-brain` and `--no-report` are given at both. The [CLI reference](../cli/index.md) lists the options of each level.

## In this guide

| Page | What it covers |
|---|---|
| [Input](input.md) | the five input types, the layout of a matrix or time series folder, the file formats and the node table |
| [Participants, sessions and runs](selection.md) | selecting participants, sessions and tasks, BIDS filter files, and participants with several runs |
| [EEG and MEG](eeg-meg.md) | connectivity files from MNE-Connectivity, the supported methods and one result folder per frequency band |
| [Output files](output.md) | the folders and tables each level writes, the comparison tables, and what happens when something fails |
| [Reports](reports.md) | how the reports are built and rebuilt, their figures, the brain template and memory use |

```{toctree}
:hidden:

input
selection
eeg-meg
output
reports
```
