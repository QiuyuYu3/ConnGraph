# ConnGraph

Graph-theory metrics, group comparisons and network-based statistics for brain connectivity from fMRI, fNIRS, EEG and MEG, with HTML reports and 3-D brain figures.

```{image} images/banner.png
:alt: A brain network on the brain, as a bundled circos plot and in a spring layout
```

:::{note}
This package is in early development; the options may change.
:::

## What it does

| Step | |
|---|---|
| Read | XCP-D (fMRI), NIRSPipe (fNIRS) and MNE-Connectivity (EEG, MEG) outputs as they are, and connectivity matrices or regional time series from any other pipeline in common table, NumPy, MATLAB, CIFTI and AFNI formats |
| Build graphs | one graph per participant with one of twelve methods, such as TMFG, density thresholding or orthogonal minimum spanning trees |
| Compute metrics | node, network and whole-graph metrics with bctpy, the Python Brain Connectivity Toolbox |
| Compare groups | the metrics, the network blocks and every edge, with covariates and permutation-based corrections, and the network-based statistic |
| Report | an HTML report for each participant and for the group, with a Methods section written from the run's settings |

`conngraph` works like a BIDS App: a participant level, then a group level. Everything it does is also available from Python.

```{toctree}
:hidden:

getting-started
guide
auto_examples/index
cli
```
