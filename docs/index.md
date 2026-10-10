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

## Where to go

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Getting started
:link: getting-started
:link-type: doc
Install the package, see what it downloads on first use, and run both levels on 20 simulated participants.
:::

:::{grid-item-card} Gallery
:link: auto_examples/index
:link-type: doc
Python examples on simulated data, run when the site is built, each with its figures and a downloadable script and notebook.
:::

:::{grid-item-card} User guide
:link: guide/index
:link-type: doc
The two levels, input types and file formats, participants, sessions and runs, EEG and MEG, output files and reports.
:::

:::{grid-item-card} CLI reference
:link: cli/index
:link-type: doc
Every option of the `conngraph` command, generated from the command itself: input, both levels, participant level and group level.
:::
::::

```{toctree}
:hidden:

getting-started
guide/index
auto_examples/index
cli/index
```
