# EEG and MEG

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

## Methods and bands

- The method is read from the files and named in the report. Undirected methods are supported: `coh`, `imcoh` (taken as its absolute value), `plv`, `ciplv`, `ppc`, `pli`, `pli2_unbiased`, `wpli`, `wpli2_debiased` and envelope correlation. Directed methods such as `dpli`, `psi` or Granger causality are refused.
- Envelope correlation is Fisher z-transformed when averaged and compared, like other correlations; the phase and coherence measures are used as they are.
- Each frequency band averaged with `faverage=True` gets its own result folder, such as `band-8to13Hz`; files with single frequencies are refused. Envelope correlation has no band axis, so band-pass the data before computing it, and average its per-epoch matrices with `conn.combine()` before saving.

## Nodes

- Node names come from the files. `nodes.tsv` adds networks (lobes or regions, for example), hemispheres and coordinates; without a network column only the node level is computed.
- Brain figures need x, y, z in the template's space. Regions of a source-space atlas have them; scalp electrode positions lie outside the brain surface.

:::{important}
The phase-locking value and coherence are biased upward when there are few epochs. If epoch counts differ between groups, prefer a debiased measure such as `wpli2_debiased` or `ppc`; the report gives the range of epoch counts.
:::

Matrices of these measures computed elsewhere can be given as `matrix` input with `--no-fisher-z`.
