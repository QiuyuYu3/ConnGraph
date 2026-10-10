# ConnGraph

Graph-theory metrics, group comparisons and network-based statistics for brain connectivity from fMRI, fNIRS, EEG and MEG, with HTML reports and 3-D brain figures.

![A brain network on the brain, as a bundled circos plot and in a spring layout](https://raw.githubusercontent.com/QiuyuYu3/brainnet3d/main/docs/images/banner.png)

- Reads XCP-D (fMRI), NIRSPipe (fNIRS) and MNE-Connectivity (EEG, MEG) outputs as they are, each frequency band of the latter on its own, and connectivity matrices or regional time series from any other pipeline in common table, NumPy, MATLAB, CIFTI and AFNI formats.
- Builds a graph for each participant with one of twelve methods, such as TMFG, density thresholding or orthogonal minimum spanning trees, and computes node, network and whole-graph metrics with bctpy, the Python Brain Connectivity Toolbox.
- Compares two groups on the metrics, the network blocks and every edge, with covariates and permutation-based corrections, and runs the network-based statistic.
- Writes an HTML report for each participant and for the group, with a Methods section written from the run's settings.

This package is in early development; the options may change.

## Installation

Clone this repository, then from its root:

```bash
pip install .
```

## Quick start

`conngraph` works like a BIDS App: a participant level, then a group level.

```bash
conngraph derivatives/xcpd results participant \
    --input-type xcpd \
    --atlases Gordon \
    --graph-method tmfg

conngraph derivatives/xcpd results group \
    --input-type xcpd \
    --atlases Gordon \
    --groups participants.tsv \
    --group-column group \
    --contrast PT HC \
    --covariates age sex \
    --correction fdr \
    --nbs-thresh 3.5
```

Input formats, options and the output layout are described in the [user guide](https://github.com/QiuyuYu3/brainnet3d/tree/main/docs/guide); `conngraph --help` lists every option.

## Gallery

All figures below come from a mock dataset of 30 participants on the Gordon atlas.

<p align="center"><img src="https://raw.githubusercontent.com/QiuyuYu3/brainnet3d/main/docs/images/brain_strength.png" alt="Node strength on the brain" width="100%"></p>
<p align="center"><em>Group mean node strength on the brain</em></p>

<p align="center"><img src="https://raw.githubusercontent.com/QiuyuYu3/brainnet3d/main/docs/images/nbs_brain.png" alt="Group difference on the brain" width="100%"></p>
<p align="center"><em>Network-based statistic: the edges that differ between groups</em></p>

<p align="center"><img src="https://raw.githubusercontent.com/QiuyuYu3/brainnet3d/main/docs/images/fc_participant.png" alt="One participant's connectivity matrix" width="49%"> <img src="https://raw.githubusercontent.com/QiuyuYu3/brainnet3d/main/docs/images/fc_group_network_hemi.png" alt="Group mean connectivity by network and hemisphere" width="49%"></p>
<p align="center"><em>Connectivity matrices: one participant by region (left) and the group mean by network and hemisphere (right)</em></p>

<p align="center"><img src="https://raw.githubusercontent.com/QiuyuYu3/brainnet3d/main/docs/images/nbs_matrices.png" alt="Group means and difference" width="100%"></p>
<p align="center"><em>Group mean matrices and their difference</em></p>

<p align="center"><img src="https://raw.githubusercontent.com/QiuyuYu3/brainnet3d/main/docs/images/circos_bundled.png" alt="Circos plot bundled by network" width="49%"> <img src="https://raw.githubusercontent.com/QiuyuYu3/brainnet3d/main/docs/images/spring_plain.png" alt="Spring layout" width="49%"></p>
<p align="center"><em>The group network: edges bundled through their networks (left) and a spring layout (right)</em></p>

<p align="center"><img src="https://raw.githubusercontent.com/QiuyuYu3/brainnet3d/main/docs/images/report_boxplot.png" alt="Metric distributions in the report" width="100%"></p>
<p align="center"><em>Group report: a metric across network nodes, one point per participant</em></p>

<p align="center"><img src="https://raw.githubusercontent.com/QiuyuYu3/brainnet3d/main/docs/images/compare_edges.png" alt="Edge-wise t values" width="60%"></p>
<p align="center"><em>Group comparison report: t of every edge, significant cells at full colour</em></p>
