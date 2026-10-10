# Getting started

Install the package, then run both levels on simulated data to see what `conngraph` writes.

## Installation

`conngraph` needs Python 3.10 or later. Clone the repository, then install it from its root and check that the command is found:

```bash
git clone https://github.com/QiuyuYu3/brainnet3d.git
cd brainnet3d
pip install .
conngraph --version
```

### Data downloaded on first use

| Data | Used for | Cached in |
|---|---|---|
| Gordon 333-parcel node table, from the official release | node coordinates for XCP-D input with the Gordon atlas, and the simulated data below and in the gallery | `~/conngraph_data`, or the folder in `CONNGRAPH_DATA` |
| fsLR 32k midthickness surfaces, through TemplateFlow | the brain in the figures | TemplateFlow's cache, or the folder in `TEMPLATEFLOW_HOME` |

A brain volume given to `--surfaces` is turned into a surface once and cached under `surfaces/` in the `conngraph` data folder.

:::{tip}
When compute nodes have no internet access, set `CONNGRAPH_DATA` and `TEMPLATEFLOW_HOME` to a shared folder and run once on a machine that has it, such as a login node.
:::

## Quick start

The steps below use 20 simulated participants in two groups, on the Gordon atlas.

### 1. Make an input folder

```python
from conngraph.datasets import make_mock_dataset

make_mock_dataset(n_per_group=10, out_dir="mock_input")
```

This writes a `matrix` input folder: `nodes.tsv`, one `sub-<label>_matrix.tsv` per participant, and `participants.tsv` with each participant's group (`A` or `B`) and age. Group B is more connected within the Default network and less within the Visual network.

### 2. Participant level

```bash
conngraph mock_input results participant \
    --input-type matrix \
    --graph-method tmfg
```

This builds a TMFG graph from each matrix, computes the default metrics at the node and network levels, and writes a report per participant. Most of the time goes into the reports; `--no-report` skips them.

### 3. Group level

```bash
conngraph mock_input results group \
    --input-type matrix \
    --groups mock_input/participants.tsv \
    --group-column group \
    --contrast B A \
    --correction fdr \
    --nbs-thresh 3.0
```

This collects the participant results, compares group B with group A on the metrics and on the connectivity within and between networks, and runs the network-based statistic. Add `--covariates age` to adjust the comparisons, though not NBS, for age.

### 4. Open the reports

```text
results/
    sub-01.html ... sub-20.html       one report per participant
    group/
        graph_report.html             the group's graph metrics
        compare/compare_report.html   the comparisons of B with A
        nbs/nbs_report.html           the network-based statistic
```

The tables behind each report sit beside it and are linked from it; [Output files](guide/output.md) lists them. To run on your own data, see [Input](guide/input.md) and the [CLI reference](cli/index.md). The [gallery](auto_examples/index.rst) does the same steps in Python.
