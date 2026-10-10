# Output files

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

## Comparison tables

Each `compare/` table has one row per test with the t statistic (positive when the first group of `--contrast` is higher), its p-value, FDR-corrected and family-wise (max-T, `--n-perms` permutations) p-values, the group means and sizes, and whether the result is significant under the `--correction` you chose (FDR, family-wise or uncorrected; required, with `--alpha`, default 0.05).

Corrections apply within a family: one metric at one level, the whole-graph metrics of one level, the network blocks of one level, or all edges. Values missing or constant across participants are not tested.

## Missing nodes, option checks and failures

Nodes with too many missing values are dropped by looking at every participant in `INPUT`, so participants run one at a time get the same nodes as a single run. The group level checks that all participants were run with the same options. If one session or atlas fails, the others still run, and `OUTPUT/logs/` (participant) or the group folder holds `error.txt`.
