# Input

`--input-type` is required and says what `INPUT` holds:

| Type | `INPUT` |
|---|---|
| `xcpd` | an XCP-D derivatives folder; name one or more atlases with `--atlases` |
| `nirspipe` | a NIRSPipe derivatives folder; HbO and HbR are analysed separately (`--chromophore` picks one) |
| `matrix` | a folder of connectivity matrices, laid out as below |
| `timeseries` | a folder of regional time series, laid out as below; connectivity is computed first (`--connectivity`) |
| `mne` | a folder of EEG or MEG connectivity files saved by MNE-Connectivity; each frequency band is analysed separately (see [EEG and MEG](eeg-meg.md)) |

## Matrix and time series folders

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

## The node table

`nodes.tsv` has one row per node. Without a network column only the node level is computed; without x, y, z the report leaves out the brain figures. For XCP-D input with the Gordon atlas, coordinates are added automatically.
