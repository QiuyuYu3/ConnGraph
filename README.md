# ConnGraph

Graph-theory metrics, group comparisons and network-based statistics for brain connectivity, with HTML reports and 3-D brain figures.

![Node strength on the brain](https://raw.githubusercontent.com/QiuyuYu3/brainnet3d/main/docs/images/brain_strength.png)

- Reads XCP-D and fnirs-pipe outputs, or a folder of connectivity matrices or regional time series.
- Builds a graph for each participant (TMFG by default, or one of eleven other methods) and computes node, network and whole-graph metrics.
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
conngraph derivatives/xcpd results participant --input-type xcpd --atlases Gordon

conngraph derivatives/xcpd results group --input-type xcpd --atlases Gordon \
    --groups participants.tsv --group-column group --contrast PT HC --covariates age sex \
    --correction fdr --nbs-thresh 3.5
```

Input formats, options and the output layout are described in [docs/usage.md](https://github.com/QiuyuYu3/brainnet3d/blob/main/docs/usage.md); `conngraph --help` lists every option.

## Gallery

All figures below come from a mock dataset of 30 participants on the Gordon atlas.

| | |
|---|---|
| ![Group difference on the brain](https://raw.githubusercontent.com/QiuyuYu3/brainnet3d/main/docs/images/nbs_brain.png) | ![Group means and difference](https://raw.githubusercontent.com/QiuyuYu3/brainnet3d/main/docs/images/nbs_matrices.png) |
| Network-based statistic: the edges that differ between groups | Group mean matrices and their difference |
| ![Circos plot bundled by network](https://raw.githubusercontent.com/QiuyuYu3/brainnet3d/main/docs/images/circos_bundled.png) | ![Spring layout](https://raw.githubusercontent.com/QiuyuYu3/brainnet3d/main/docs/images/spring_plain.png) |
| Group network, edges bundled through their networks | Group network in a spring layout |
| ![Edge-wise t values](https://raw.githubusercontent.com/QiuyuYu3/brainnet3d/main/docs/images/compare_edges.png) | ![Metric distributions in the report](https://raw.githubusercontent.com/QiuyuYu3/brainnet3d/main/docs/images/report_boxplot.png) |
| Group comparison report: t of every edge | Group report: a metric across network nodes |

![Group network in the report](https://raw.githubusercontent.com/QiuyuYu3/brainnet3d/main/docs/images/report_group_network.png)

Group report: the group network as interactive circos plots and a spring layout; hover a region for its name.
