"""
Assemble and write the graph-metrics and NBS reports.
"""

from __future__ import annotations

import html
import json
import math
import os
import pathlib
import warnings
from functools import cache
from importlib.resources import files

import numpy as np
import pandas as pd

from conngraph.report import figures
from conngraph.report.methods import compare_methods, graph_methods, mne_details, nbs_methods, participant_count

PLOTLY_CDN = "https://cdn.plot.ly/plotly-3.5.0.min.js"
_METRIC_TITLES = {"clust_coeff": "Clustering coefficient", "btwn_cent": "Betweenness centrality",
                  "strength": "Strength", "ge_local": "Local efficiency", "eff_nodal": "Nodal efficiency",
                  "participation": "Participation coefficient", "module_z": "Within-module degree z",
                  "eig_cent": "Eigenvector centrality", "close_cent": "Closeness centrality",
                  "eff_global": "Global efficiency", "char_path": "Characteristic path length",
                  "clust_mean": "Mean clustering coefficient", "modularity": "Modularity",
                  "small_world": "Small-world index"}
_SWEEP_LABELS = {"density": "Density", "threshold": "Threshold", "alpha": "Significance level"}
_MAX_BRAIN_EDGES = 300


def save_graph_report(result, path, nodes: pd.DataFrame | None = None, surfaces: tuple[str, str] | None = None,
                      static_brain: bool = True, interactive_brain: bool = False) -> None:
    """Write the HTML report of a compute_graph_metrics result; see GraphMetricsResult.save_report."""
    params = result.params
    opts, levels = params["options"], params["levels"]
    nodes = result.nodes if nodes is None else nodes
    figs = _FigureFolder(path)
    link = _linker(result.files, path)
    label_col, network_col = opts["label_col"], opts["network_col"]
    networks = _column(nodes, label_col, network_col)
    palette = _palette(networks)
    metrics = list(dict.fromkeys(m for d in levels.values() for m in d["metrics"]))
    notes: list[str] = [f"{n} ROIs labelled “{k}” were left out of the network levels; the node level keeps them."
                        for k, n in params.get("excluded_rois", {}).items()]

    body, sections = [], []
    for level, df, corr, sid, title in (("network", result.network_df, result.net_corr_df, "Network", "Network level"),
                                        ("network_hemi", result.net_hemi_df, result.net_hemi_corr_df, "NetworkHemi",
                                         "Network level (hemispheres)")):
        if df is None:
            continue
        hemi = level == "network_hemi"
        values, connectivity = ("net_hemi_df", "net_hemi_corr_df") if hemi else ("network_df", "net_corr_df")
        net_of = (lambda n: n.partition("_")[2]) if hemi else (lambda n: n)
        names = _metric_names(df)
        boxes = [(m, _metric_title(m), figures.to_div(figures.level_boxplot(df[m], palette, _metric_title(m), hemi)))
                 for m in names]
        M = _wide_to_matrix(corr)
        zmax = float(np.nanmax(np.abs(M.values))) or 1.0
        heat = figures.ordered_heatmap(M.values, list(M.index), [net_of(n) for n in M.index], zmax,
                                       "Fisher z" if opts.get("apply_fisher_z", True) else value_label(params))
        body.append(dict(id=sid, title=title, desc=_level_desc(params, level), steps=[
            _step("a. Metric distributions", data=link(values), html=_picker(boxes),
                  desc="One box per network node" + (", left hemisphere darker than right" if hemi else "")
                       + "; each point is a participant (hover for the ID)."),
            _step("b. Connectivity", data=link(connectivity), html=figures.to_div(heat),
                  desc="Group mean connectivity within (diagonal) and between network nodes: the matrix these graphs were built from."),
        ]))
        sections.append((sid, title))

    if result.node_df is not None:
        body.append(dict(id="Node", title="Node level", desc=_level_desc(params, "node"),
                         steps=_node_steps(result, _metric_names(result.node_df), nodes, label_col, networks, palette,
                                           surfaces, static_brain, interactive_brain, notes, figs, link)))
        sections.append(("Node", "Node level"))

    if result.global_df is not None:
        body.append(dict(id="Global", title="Whole graph", desc="", steps=_global_steps(result.global_df, link)))
        sections.append(("Global", "Whole graph"))

    if result.curves is not None:
        first = levels.get("node") or next(iter(levels.values()))
        sweep = next(k for k, v in first["graph_params"].items() if isinstance(v, list))
        shown = list(dict.fromkeys(result.curves["metric"]))
        fig = figures.curves_figure(result.curves, shown, {m: _metric_title(m) for m in shown},
                                    _SWEEP_LABELS.get(sweep, sweep))
        body.append(dict(id="Curves", title="Across " + _SWEEP_LABELS.get(sweep, sweep).lower() + " values", desc="", steps=[
            _step("a. Metrics at each value", data=link("curves"), html=figures.to_div(fig),
                  desc="Mean over nodes, then mean ± SD over participants: the values the summary integrates.")]))
        sections.append(("Curves", "Across values"))

    errors = [f"{lvl} / {sid}: {msg}" for lvl, subs in result.failed.items() for sid, msg in subs.items()]
    loaded = params.get("input") or {}
    warned = params.get("warnings", []) + _input_warnings(loaded)
    errors += warned
    failed = sum(len(v) for v in result.failed.values())
    first = levels.get("node") or next(iter(levels.values()))
    notes += _input_notes(loaded)
    summary = [
        _input_row(loaded)
        + [("Participants", f"{participant_count(params)}" + (f" ({len(params['subjects'])} matrices, runs kept apart)"
                                                              if loaded.get("split_runs") else "")),
         ("Levels", "; ".join(f"{_level_title(n)} ({d['n_nodes']} nodes)" for n, d in levels.items())),
         ("Graph construction", _method_text(first)),
         ("Sign rule", first["sign"]),
         ("Metrics", ", ".join(metrics))],
        [("Created", params["created"].replace("T", " ")),
         ("ConnGraph", "v" + params["packages"]["conngraph"]),
         ("Python", params["python"]),
         ("Failed subjects", _flag(f"{failed}/{len(params['subjects'])}", failed == 0, "bad")),
         ("Warnings", _flag(len(warned), not warned, "warn"))],
    ]
    call = ("compute_graph_metrics", ["matrices", "atlas"],
            {**opts, "graph_method": first["graph_method"], "graph_params": first["graph_params"]})
    _write(path, "Graph metrics report", params, sections, summary, call, body, errors, notes, graph_methods(params),
           f"{participant_count(params)} participants")


def save_nbs_report(result, path, nodes: pd.DataFrame | None = None, surfaces: tuple[str, str] | None = None,
                    static_brain: bool = True, label_col: str = "label", network_col: str = "network",
                    interactive_brain: bool = False) -> None:
    """Write the HTML report of a run_nbs result; see NBSResult.save_report."""
    params, o = result.params, result.params["options"]
    figs = _FigureFolder(path)
    link = _linker(result.files, path)
    labels = list(result.labels) if result.labels is not None else [str(i) for i in range(len(result.adj))]
    networks = _column(nodes, label_col, network_col)
    nets = [networks.get(lab, "None") for lab in labels] if networks is not None else None
    palette = _palette(networks)
    k = o["k"]
    sizes = np.array([(result.adj == i + 1).sum() / 2 for i in range(len(result.pval))])
    sig = [i for i, p in enumerate(result.pval) if p < 0.05]
    adj_sig = np.isin(result.adj, np.array(sig) + 1) if sig else np.zeros_like(result.adj, dtype=bool)
    diff = result.mean_g1 - result.mean_g2
    notes: list[str] = []

    comp = pd.DataFrame({"Component": np.arange(1, len(result.pval) + 1), "Edges": sizes.astype(int),
                         "Nodes": [int(np.any(result.adj == i + 1, axis=0).sum()) for i in range(len(result.pval))],
                         "p": [_fmt_p(p, k) for p in result.pval],
                         "Result": ['<span class="ok">significant</span>' if p < 0.05 else "n.s." for p in result.pval]})
    comp = comp.iloc[np.argsort(result.pval, kind="stable")]
    shown = sig + [i for i in np.argsort(-sizes) if result.pval[i] >= 0.05][:1]
    hist = figures.null_histogram(result.null, sizes[shown], [f"C{i + 1}, p {_p_relation(result.pval[i], k)}" for i in shown],
                                  [result.pval[i] < 0.05 for i in shown])
    body = [dict(id="Components", title="Components", desc="", steps=[
        _step("a. Components above threshold", data=link("components"), html=_table(comp, sortable=False),
              desc="Connected components of suprathreshold edges, by p-value."),
        _step("b. Null distribution", data=link("null"), html=figures.to_div(hist),
              desc="Largest component size in each permutation; vertical lines mark observed components."),
    ])]
    sections = [("Components", "Components")]

    contrast = (params.get("input") or {}).get("contrast") or ["", ""]
    g1, g2 = contrast if contrast[0] else ("Group 1", "Group 2")
    if sig:
        steps = []
        letter = iter("abcdefgh")
        iu = [tuple(e) for e in np.argwhere(np.triu(adj_sig, 1))]
        n_sig = len(iu)
        if n_sig > _MAX_BRAIN_EDGES:
            order = np.argsort([-abs(diff[i, j]) for i, j in iu], kind="stable")
            drawn = [iu[t] for t in order[:_MAX_BRAIN_EDGES]]
            what = f"the {_MAX_BRAIN_EDGES} edges with the largest difference out of {n_sig} significant edges"
        else:
            drawn, what = iu, f"all {n_sig} significant edges"
        xyz = _coordinates(nodes, label_col, labels)
        if xyz is not None and (static_brain or interactive_brain):
            degree = adj_sig.sum(axis=1)
            html_ = ""
            if static_brain:
                html_ += _static_nbs_brain(nodes, label_col, network_col, labels, drawn, diff, degree, palette,
                                           surfaces, figs)
            if interactive_brain:
                html_ += figs.save_html(figures.brain_edges(
                    xyz, labels, nets or ["None"] * len(labels), drawn, np.array([diff[i, j] for i, j in drawn]),
                    degree, palette, figures.surface_meshes(_surfaces(surfaces)),
                    ("Group 1 > Group 2", "Group 1 < Group 2")), "brain_edges")
            steps.append(_step(f"{next(letter)}. On the brain", data=link("edges"), html=html_,
                               desc=f"Showing {what}, coloured by the sign of the group difference (group 1 − group 2); "
                                    "node size is the number of significant edges at each region."))
        elif xyz is None:
            notes.append(_no_coordinates(nodes))
        if nets:
            import networkx as nx

            G = nx.Graph()
            G.add_nodes_from(range(len(labels)))
            G.add_weighted_edges_from((int(i), int(j), diff[i, j]) for i, j in iu)
            steps.append(_step(f"{next(letter)}. On a circle", data=link("edges"), html=_figure_row(*_circos_images(G, labels, nets, palette, f"{g1} − {g2}")),
                               desc=f"All {n_sig} significant edges with regions grouped by network: coloured by the "
                                    "group difference, then bundled through their networks and coloured by the "
                                    "networks they join."))
        steps.append(_step(f"{next(letter)}. Group difference and means", data=link("means"),
                           html=_nbs_matrices(result, adj_sig, labels, nets, (g1, g2), diff),
                           desc="Group 1 minus group 2, then the mean connectivity of each group"
                                + (", ordered by network" if nets else "")
                                + "; significant edges keep their colour and the others are faded."))
        if nets:
            names = sorted(set(nets))
            names = [names[i] for i in figures.network_order(names, names)]
            counts = pd.DataFrame(0, index=names, columns=names)
            for i, j in iu:
                counts.loc[nets[i], nets[j]] += 1
                if nets[i] != nets[j]:
                    counts.loc[nets[j], nets[i]] += 1
            keep = [n for n in names if counts.loc[n].sum() > 0]
            steps.append(_step(f"{next(letter)}. By network pair", data=link("edges"), html=figures.to_div(figures.count_heatmap(counts.loc[keep, keep])),
                               desc="Number of significant edges within and between networks."))
        body.append(dict(id="Edges", title="Significant edges",
                         desc=f"{n_sig} significant edges, each listed in nbs_edges.tsv.", steps=steps))
        sections.append(("Edges", "Significant edges"))

    loaded = params.get("input") or {}
    notes += _input_notes(loaded)
    names = loaded.get("contrast") or ["", ""]
    summary = [
        _input_row(loaded)
        + [("Group 1", f"{names[0] + ': ' if names[0] else ''}{len(params['groups']['g1'])} participants"),
         ("Group 2", f"{names[1] + ': ' if names[1] else ''}{len(params['groups']['g2'])} participants"),
         ("Test", "paired t-test" if o["paired"] else "two-sample t-test"),
         ("Threshold", f"t > {o['thresh']} ({o['tail']} tail)"),
         ("Permutations", f"{k} (seed {o['seed']})")],
        [("Created", params["created"].replace("T", " ")),
         ("ConnGraph", "v" + params["packages"]["conngraph"])],
    ]
    quantities = pd.DataFrame([("Components", str(len(result.pval))),
                               ("Significant (p < 0.05)", _flag(len(sig), bool(sig), "")),
                               ("Smallest p", _fmt_p(min(result.pval), k) if len(result.pval) else "–")],
                              columns=["Quantity", "Value"])
    _write(path, "Network-based statistic report", params, sections, summary, ("run_nbs", ["matrices_g1", "matrices_g2"], o),
           body, _input_warnings(loaded), notes, nbs_methods(params), f"{len(params['groups']['g1'])} vs {len(params['groups']['g2'])}",
           quantities=_table(quantities, sortable=False))


def save_compare_report(result, path, nodes: pd.DataFrame | None = None, label_col: str = "label",
                        network_col: str = "network") -> None:
    """Write the HTML report of a compare_groups result; see GroupComparisonResult.save_report."""
    params, o = result.params, result.params["options"]
    if o.get("correction") is None:
        raise ValueError("the report marks significant results, so compare_groups needs a correction: fdr, fwe or none")
    link = _linker(result.files, path)
    networks = _column(nodes, label_col, network_col)
    g1, g2 = params["contrast"]
    kind = _CORRECTION_P[o["correction"]]
    alpha = o["alpha"]
    body, sections, counts = [], [], []
    for section, prefix in (("Metrics", "metrics_"), ("Global", "global_"), ("Blocks", "blocks_"), ("Edges", "edges")):
        names = [n for n in result.tables if n.startswith(prefix)]
        if not names:
            continue
        steps, letter = [], iter("abcdefgh")
        for name in names:
            table = result.tables[name]
            sig = table[table["significant"]].sort_values(kind)
            tested = int(table['t'].notna().sum())
            counts.append((_COMPARE_TITLES[name], _flag(f"{len(sig)} of {tested}", True, "")))
            if name == "edges":
                # too many rows to read on the page; the heatmap shows them and the file lists them
                desc = (f"{len(sig)} of {tested} edges significant at {_P_NAMES[o['correction']]} p < {alpha}; "
                        f"every test is listed in {name}.tsv.")
                steps.append(_step(f"{next(letter)}. {_COMPARE_TITLES[name]}", data=link(name),
                                   html=_compare_heatmap(name, table, kind, networks), desc=desc))
                continue
            if len(sig):
                shown = sig.head(_MAX_COMPARE_ROWS)
                desc = (f"{len(sig)} significant at {_P_NAMES[o['correction']]} p < {alpha}, smallest p first"
                        + (f"; the first {_MAX_COMPARE_ROWS} are shown, every test is listed in {name}.tsv"
                           if len(sig) > _MAX_COMPARE_ROWS else "") + ".")
            else:
                shown = table[table["t"].notna()].sort_values(kind).head(10)
                desc = f"No test reached {_P_NAMES[o['correction']]} p < {alpha}; the ten smallest p-values are shown."
            html_ = _table(_compare_rows(shown, networks, g1, g2))
            heat = _compare_heatmap(name, table, kind, networks)
            if heat:
                html_ += heat
            steps.append(_step(f"{next(letter)}. {_COMPARE_TITLES[name]}", data=link(name), html=html_, desc=desc))
        body.append(dict(id=section, title=_COMPARE_SECTIONS[section], desc="", steps=steps))
        sections.append((section, _COMPARE_SECTIONS[section]))

    loaded = params.get("input") or {}
    notes = _input_notes(loaded)
    for name, items in (params.get("untested") or {}).items():
        notes.append(f"{len(items)} test{'' if len(items) == 1 else 's'} in {name} had missing or constant values and "
                     f"{'was' if len(items) == 1 else 'were'} not run: {', '.join(items[:10])}"
                     + (" …" if len(items) > 10 else "") + ".")
    errors = _input_warnings(loaded)
    left = params.get("left_out") or {}
    if left.get("no_data"):
        errors.append(_participants(len(left["no_data"]), "had no data for every comparison and {} left out")
                      + ": " + ", ".join(map(str, left["no_data"])) + ".")
    if left.get("covariates"):
        errors.append(_participants(len(left["covariates"]), "had a missing covariate and {} left out")
                      + ": " + ", ".join(map(str, left["covariates"])) + ".")
    covariates = ", ".join(o["covariates"]) or "none"
    summary = [
        _input_row(loaded)
        + [("Group 1", f"{g1}: {len(params['groups']['g1'])} participants"),
           ("Group 2", f"{g2}: {len(params['groups']['g2'])} participants"),
           ("Covariates", covariates),
           ("Significance", f"{_P_NAMES[o['correction']]} p < {alpha}"),
           ("Permutations", f"{o['n_perms']} (seed {o['seed']})")],
        [("Created", params["created"].replace("T", " ")),
         ("ConnGraph", "v" + params["packages"]["conngraph"])],
    ]
    quantities = pd.DataFrame(counts, columns=["Tests", "Significant"])
    _write(path, "Group comparison report", params, sections, summary, ("compare_groups", ["groups", "contrast"], o),
           body, errors, notes, compare_methods(params), f"{g1} vs {g2}", quantities=_table(quantities, sortable=False))


_CORRECTION_P = {"fdr": "p_fdr", "fwe": "p_fwe", "none": "p"}
_P_NAMES = {"fdr": "FDR-corrected", "fwe": "family-wise corrected", "none": "uncorrected"}
_COMPARE_SECTIONS = {"Metrics": "Graph metrics", "Global": "Whole graph", "Blocks": "Network blocks", "Edges": "Edges"}
_COMPARE_TITLES = {"metrics_node": "Node level", "metrics_network": "Network level",
                   "metrics_networkhemi": "Network level (hemispheres)", "global_node": "Node-level graph",
                   "global_network": "Network-level graph", "global_networkhemi": "Network-level graph (hemispheres)",
                   "blocks_network": "Between networks", "blocks_networkhemi": "Between networks (hemispheres)",
                   "edges": "Every edge"}
_MAX_COMPARE_ROWS = 10


def _compare_rows(table: pd.DataFrame, networks: dict | None, g1: str, g2: str) -> pd.DataFrame:
    out = pd.DataFrame(index=table.index)
    if "metric" in table:
        out["Metric"] = [_metric_title(m) for m in table["metric"]]
    for col, title in (("node", "Node"), ("network", "Network"), ("network_a", "Network A"), ("network_b", "Network B"),
                       ("roi_a", "ROI A"), ("roi_b", "ROI B")):
        if col in table:
            out[title] = table[col].astype(str)
    if networks is not None and "node" in table:
        out["Network"] = [networks.get(n, "None") for n in table["node"]]
    if networks is not None and "roi_a" in table:
        out["Network A"] = [networks.get(n, "None") for n in table["roi_a"]]
        out["Network B"] = [networks.get(n, "None") for n in table["roi_b"]]
    for col, title in (("t", "t"), ("p", "p"), ("p_fdr", "p FDR"), ("p_fwe", "p FWE"), ("mean_group1", g1),
                       ("mean_group2", g2)):
        out[title] = table[col].astype(float)
    return out.reset_index(drop=True)


def _compare_heatmap(name: str, table: pd.DataFrame, kind: str, networks: dict | None) -> str:
    """t of every block or edge as a matrix, the significant ones at full colour."""
    if name.startswith("blocks_"):
        a, b, groups_of = "network_a", "network_b", (lambda n: n.partition("_")[2] if name.endswith("hemi") else n)
    elif name == "edges":
        a, b, groups_of = "roi_a", "roi_b", (lambda n: networks.get(n, "None")) if networks is not None else None
    else:
        return ""
    names = list(dict.fromkeys([*table[a].astype(str), *table[b].astype(str)]))
    pos = {n: k for k, n in enumerate(names)}
    M = np.zeros((len(names), len(names)))
    for x, y, t in zip(table[a].astype(str), table[b].astype(str), table["t"].fillna(0.0)):
        M[pos[x], pos[y]] = M[pos[y], pos[x]] = t
    sig = table[table["significant"]]
    marks = [p for x, y in zip(sig[a].astype(str), sig[b].astype(str)) for p in ((x, y), (y, x))]
    groups = [groups_of(n) for n in names] if groups_of else None
    fig = figures.ordered_heatmap(M, names, groups, float(np.abs(M).max()) or 1.0, "t", marks or None)
    label = "t, significant cells at full colour" if marks else "t"
    return f'<div class="option-label">{label}</div>' + figures.to_div(fig)


def _node_steps(result, metrics, nodes, label_col, networks, palette, surfaces, static_brain, interactive_brain,
                notes, figs, link) -> list[dict]:
    steps = []
    labels = list(result.node_df.columns.get_level_values(1).unique())
    xyz = _coordinates(nodes, label_col, labels)
    if xyz is not None and (static_brain or interactive_brain):
        keep = ~np.isnan(xyz).any(axis=1)
        meshes = figures.surface_meshes(_surfaces(surfaces)) if interactive_brain else []
        nets = [networks.get(lab, "None") if networks is not None else "" for lab in labels]
        panes = []
        for m in metrics:
            values = result.node_df[m].mean(axis=0).reindex(labels).to_numpy(float)
            t = _metric_title(m)
            shown = keep & ~np.isnan(values)
            if not shown.any():
                panes.append((m, t, NO_VALUES))
                continue
            html_ = ""
            if static_brain:
                html_ += _static_node_brain(result, nodes, label_col, labels, values, t, surfaces, figs, m)
            if interactive_brain:
                html_ += figs.save_html(figures.brain_values(
                    xyz[shown], [lab for lab, k in zip(labels, shown) if k], [n for n, k in zip(nets, shown) if k],
                    values[shown], t, meshes), f"brain_{m}")
            panes.append((m, t, html_))
        steps.append(_step("a. Group mean on the brain", data=link("node_df"), html=_picker(panes, columns=1),
                           desc="Colour and size both show the mean over participants."
                                + (" Each link opens a view that can be rotated, with region names on hover."
                                   if interactive_brain else "")))
    elif xyz is None:
        notes.append(_no_coordinates(nodes))
    letter = iter("abcdef"[len(steps):])
    if networks is not None:
        boxes = []
        for m in metrics:
            values = result.node_df[m].mean(axis=0)
            nets = pd.Series([networks.get(lab, "None") for lab in values.index], index=values.index)
            boxes.append((m, _metric_title(m), figures.to_div(figures.node_boxplot(values, nets, palette, _metric_title(m)))))
        steps.append(_step(f"{next(letter)}. Values by network", data=link("node_df"), html=_picker(boxes, columns=1),
                           desc="Mean over participants of each region; hover for its name."))
    tops = []
    for m in metrics:
        top = result.node_df[m].mean(axis=0).sort_values(ascending=False).head(15)
        table = pd.DataFrame({"ROI": top.index})
        if networks is not None:
            table["Network"] = [networks.get(lab, "None") for lab in top.index]
        table["Group mean"] = top.values
        tops.append((m, _metric_title(m), _table(table)))
    steps.append(_step(f"{next(letter)}. Highest regions", data=link("node_df"), html=_picker(tops, columns=3),
                       desc="The 15 regions with the highest mean over participants."))
    if result.mean_matrix is not None:
        M = result.mean_matrix
        names = list(M.index)
        groups = [networks.get(n, "None") for n in names] if networks is not None else None
        steps.append(_step(f"{next(letter)}. Connectivity", data=link("mean_matrix"), html=figures.to_div(figures.ordered_heatmap(
            M.to_numpy(float), names, groups, 1.0, value_label(result.params))),
            desc="Group mean connectivity between all regions" + (", ordered by network" if groups else "") + "."))
    group = _group_graph(result) if networks is not None else None
    if group is not None:
        G, names, how = group
        nets = [networks.get(n, "None") for n in names]
        spring = figures.to_div(figures.spring_figure(G, names, nets, palette))
        steps.append(_step(f"{next(letter)}. Group network", data=link("mean_matrix"),
                           html=_figure_row(*_circos_images(G, names, nets, palette, f"group mean {value_label(result.params)}"))
                                + _figure_block("Spring layout", spring),
                           desc=f"The group mean connectivity turned into a graph the way each participant's was "
                                f"({how}); the metrics above come from each participant's own graph. The circle "
                                "groups regions by network; the bundled version routes edges through their networks "
                                "and colours them by the networks they join. Hover a region for its name."))
    return steps


def _group_graph(result):
    """The group mean matrix built into a graph with the node level's method, or None when it cannot be redone."""
    if result.mean_matrix is None:
        return None
    opts = result.params["options"]
    return _rebuilt_graph(result.mean_matrix, result.params, opts.get("apply_fisher_z", True))


def _rebuilt_graph(matrix: pd.DataFrame, params: dict, is_z: bool):
    """A matrix built into a graph with the node level's method, or None when it cannot be redone."""
    import networkx as nx

    from conngraph.graph_theory.sparsify import GRAPH_METHODS, apply_sign, build_adjacency

    level = params["levels"].get("node") or {}
    method = level.get("graph_method")
    # a custom function cannot be rebuilt from its name, and PMFG takes minutes on a full atlas
    if method not in GRAPH_METHODS or method == "pmfg":
        return None
    graph_params, how = dict(level.get("graph_params") or {}), _method_text(level)
    for key, value in graph_params.items():
        if isinstance(value, list):
            graph_params[key] = value[len(value) // 2]
            how = f"{level['graph_method']}, {key} {graph_params[key]} from the middle of the range"
    M = np.nan_to_num(matrix.to_numpy(float))
    if is_z:
        M = np.tanh(M)
    np.fill_diagonal(M, 0)
    M = apply_sign(M, level["sign"])
    if params["options"].get("normalize_weights") and np.abs(M).max() > 0:
        M = M / np.abs(M).max()
    A = build_adjacency(M, method, graph_params, level["sign"] == "signed")
    return nx.from_numpy_array(A), list(matrix.index), f"{how}, {level['sign']} weights"


def value_label(params: dict) -> str:
    """Colour bar name of the connectivity values: r, or the measure read from MNE-Connectivity files."""
    return (params.get("input") or {}).get("measure_label") or "r"


def _circos_images(G, labels, nets, palette, colorbar_title) -> list[str]:
    return [_figure_block("Circos", figures.to_div(figures.circos_figure(G, labels, nets, palette, colorbar_title))),
            _figure_block("Circos, bundled by network", figures.to_div(
                figures.circos_figure(G, labels, nets, palette, colorbar_title, bundled=True)))]


def _nbs_matrices(result, adj_sig, labels, nets, group_names, diff) -> str:
    off = ~np.eye(len(labels), dtype=bool)
    lim = float(max(np.abs(result.mean_g1[off]).max(), np.abs(result.mean_g2[off]).max())) or 1.0
    marks = [(labels[i], labels[j]) for i, j in np.argwhere(adj_sig)]
    panes = [(f"g{k}", name, figures.to_div(figures.ordered_heatmap(M, labels, nets, lim, name, marks)))
             for k, (name, M) in enumerate(zip(group_names, (result.mean_g1, result.mean_g2)), 1)]
    name = " − ".join(group_names)
    # the difference gets a row of its own, the two means share the next
    return _figure_block(name, figures.to_div(figures.ordered_heatmap(
        diff, labels, nets, float(np.abs(diff).max()) or 1.0, name, marks))) + _picker(panes)


def _figure_block(label: str, img: str) -> str:
    return f'<div class="option-label">{html.escape(label)}</div>' + img


def _figure_row(*blocks: str) -> str:
    """Figures side by side, one under another in a narrow window."""
    # the column count is fixed up front: plotly sizes each figure as it is parsed, before its neighbours exist
    return (f'<div class="figure-row" style="--columns:{len(blocks)}">' + "".join(f"<div>{b}</div>" for b in blocks)
            + "</div>")


def _static_node_brain(result, nodes, label_col, labels, values, title, surfaces, figs, metric) -> str:
    import conngraph as bnv

    # a region without a value (e.g. no network for a partition-based metric) would blank the whole render
    shown = [lab for lab, v in zip(labels, values) if not np.isnan(v)]
    nd = nodes.rename(columns={label_col: "label"}).copy()
    nd[title] = nd["label"].map(dict(zip(labels, values)))
    nd = nd[nd["label"].isin(shown)]
    M = (result.mean_matrix.loc[shown, shown] if result.mean_matrix is not None
         else pd.DataFrame(0.0, index=shown, columns=shown))
    left, right = _surfaces(surfaces) or (None, None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds = bnv.load(M, nd)
        fig = bnv.BrainNetPlotter(ds).plot_views(
            views=[{"view": "L"}, {"view": "S"}, {"view": "R"}], node_color=title, node_size=title, node_cmap="viridis",
            node_size_range=(1.5, 6.0), edge_threshold=2.0, surface_L=left, surface_R=right, surface_alpha=0.12,
            legend=["node_color"], width=10.5)
    # rendered panels already sit at their own resolution; resampling them adds bytes, not detail
    return figs.save(fig, f"brain_{metric}", dpi="figure")


def _static_nbs_brain(nodes, label_col, network_col, labels, drawn, diff, degree, palette, surfaces, figs) -> str:
    import conngraph as bnv

    nd = nodes.rename(columns={label_col: "label", network_col: "network"}).copy()
    nd = nd[nd["label"].isin(labels)]
    nd["network"] = nd["label"].map(_column(nd, "label", "network"))
    nd["Significant edges"] = nd["label"].map(dict(zip(labels, degree)))
    M = np.zeros_like(diff)
    for i, j in drawn:
        M[i, j] = M[j, i] = diff[i, j]
    left, right = _surfaces(surfaces) or (None, None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds = bnv.load(pd.DataFrame(M, index=labels, columns=labels), nd)
        fig = bnv.BrainNetPlotter(ds).plot_views(
            views=[{"view": "L"}, {"view": "S"}, {"view": "R"}], node_color="network", node_size="Significant edges",
            node_size_range=(0.8, 6.0), edge_threshold=1e-12, edge_color="weight", surface_L=left, surface_R=right,
            surface_alpha=0.12, legend=["node_color", "edge_color"], legend_titles={"edge_color": "Group 1 − Group 2"},
            node_palette=palette or None, width=10.5)
    return figs.save(fig, "brain_edges", dpi="figure")


def _write(path, title, params, sections, summary, call, body, errors, notes, methods, chip,
           quantities: str = "") -> None:
    command = params.get("command")
    fn, args, opts = call
    call_text = command or f"{fn}(\n    " + ",\n    ".join(args + [f"{k}={json.dumps(v)}" for k, v in opts.items()]) + ",\n)"
    page = _environment().get_template("report.html.j2").render(
        page_title=title, heading=title, version=params["packages"]["conngraph"],
        created=params["created"].replace("T", " "), chip=chip, sections=sections, summary=summary, quantities=quantities,
        call_title="Run command" if command else "Call", call=html.escape(call_text), body=body,
        errors=[html.escape(e) for e in errors], notes=notes, methods=methods,
        versions={"python": params["python"], **{k: v for k, v in params["packages"].items() if v}},
        css=files("conngraph.report").joinpath("templates/report.css").read_text(encoding="utf-8"),
        plotly_src=PLOTLY_CDN,
    )
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(page)


@cache
def _environment():
    from jinja2 import Environment, PackageLoader

    return Environment(loader=PackageLoader("conngraph.report", "templates"), autoescape=False)


def _step(title: str, html: str, desc: str = "", data: list[str] = ()) -> dict:
    return dict(title=title, html=html, desc=desc, data=list(data))


def _linker(files: dict, report_path):
    """link(*keys): the result files behind a step, relative to the report; none for a result never written."""
    base = os.path.dirname(os.path.abspath(report_path))

    def link(*keys: str) -> list[str]:
        return [os.path.relpath(p, base).replace(os.sep, "/") for k in keys for p in files.get(k, [])]
    return link


def _picker(panes: list[tuple[str, str, str]], columns: int = 2) -> str:
    """Every pane in a grid, each under its label; panes are (key, label, html)."""
    if len(panes) == 1:
        return panes[0][2]
    cells = "".join(f"<div>{_figure_block(t, h)}</div>" for _, t, h in panes)
    return f'<div class="figure-row" style="--columns:{min(columns, len(panes))}">{cells}</div>'


def _sig3(v: float) -> str:
    if v == 0:
        return "0"
    if not 1e-4 <= abs(v) < 1e6:
        return f"{v:.3g}"
    return f"{v:.{max(0, 2 - math.floor(math.log10(abs(v))))}f}"


def _float_format(name: str, values: pd.Series):
    """p-values to 3 decimals; else one decimal count per column, or per cell when the column spans over tenfold."""
    if name == "p" or str(name).startswith("p "):
        return lambda v: "&lt; 0.001" if v < 0.001 else f"{v:.3f}"
    finite = np.abs(values[np.isfinite(values) & (values != 0)].to_numpy(dtype=float))
    if not len(finite) or finite.max() >= 10 * finite.min() or not 1e-4 <= finite.min() < 1e6:
        return _sig3
    decimals = max(0, 2 - math.floor(math.log10(np.median(finite))))
    return lambda v: f"{v:.{decimals}f}"


def _table(df: pd.DataFrame, sortable: bool = True) -> str:
    # short headers such as "p FDR" stay on one line
    head = "".join(f'<th class="{"num" if pd.api.types.is_numeric_dtype(df[c]) else ""}">'
                   f'{html.escape(str(c)).replace(" ", "&nbsp;") if len(str(c)) <= 12 else html.escape(str(c))}</th>'
                   for c in df.columns)
    formats = {c: _float_format(c, df[c]) for c in df.columns if pd.api.types.is_float_dtype(df[c])}
    rows = []
    for row in df.itertuples(index=False):
        cells = []
        for c, v in zip(df.columns, row):
            if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float, np.integer, np.floating)):
                cells.append(f"<td>{v}</td>")
            elif isinstance(v, (int, np.integer)):
                cells.append(f'<td class="num" data-v="{v}">{v}</td>')
            elif not np.isfinite(v):
                cells.append(f'<td class="num" data-v="{v}"><span class="na">n/a</span></td>')
            else:
                cells.append(f'<td class="num" data-v="{v:.10g}">{formats.get(c, _sig3)(v)}</td>')
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f'<table class="flat{" sortable" if sortable else ""}"><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table>'


class _FigureFolder:
    """Figures saved as files (in figures/ beside the report by default), named by prefix, linked from the page."""

    def __init__(self, report_path, folder=None, prefix: str | None = None):
        path = pathlib.Path(report_path).resolve()
        self.page = path
        self.folder = pathlib.Path(folder).resolve() if folder else path.parent / "figures"
        self.stem = prefix or path.stem
        self.src = os.path.relpath(self.folder, path.parent).replace(os.sep, "/")
        # figures left from an earlier report of the same name would otherwise linger
        for old in [*self.folder.glob(f"{self.stem}_*.png"), *self.folder.glob(f"{self.stem}_*.html")]:
            old.unlink()

    def save(self, fig, name: str, dpi: float | str = 300) -> str:
        import matplotlib.pyplot as plt

        self.folder.mkdir(parents=True, exist_ok=True)
        filename = f"{self.stem}_{name}.png"
        fig.savefig(self.folder / filename, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return f'<img class="figure" src="{html.escape(self.src)}/{html.escape(filename)}" alt="">'

    def save_html(self, fig, name: str) -> str:
        self.folder.mkdir(parents=True, exist_ok=True)
        filename = f"{self.stem}_{name}.html"
        fig.write_html(self.folder / filename, include_plotlyjs="cdn", default_height="95vh",
                       config={"displaylogo": False, "responsive": True})
        return (f'<p class="figure-link"><a href="{html.escape(self.src)}/{html.escape(filename)}" target="_blank">'
                "Open the rotatable 3-D view</a></p>")


def _flag(value, ok: bool, bad_class: str) -> str:
    return f'<span class="{"ok" if ok else bad_class}">{value}</span>'


def _fmt_p(p: float, k: int) -> str:
    return f"< {1 / k:.3g}" if p == 0 else f"{p:.3g}"


def _p_relation(p: float, k: int) -> str:
    return _fmt_p(p, k) if p == 0 else f"= {_fmt_p(p, k)}"


def _metric_names(df: pd.DataFrame) -> list[str]:
    return list(df.columns.get_level_values(0).unique())


def _metric_title(name: str) -> str:
    metric, *rest = name.split(".")
    details = [*rest[:1], *(f"{p} partition" for p in rest[1:] if p != "norm"), *(["normalized"] if "norm" in rest[1:] else [])]
    base = _METRIC_TITLES.get(metric, metric)
    return f"{base} ({', '.join(details)})" if details else base


def _global_steps(df: pd.DataFrame, link) -> list[dict]:
    names = list(dict.fromkeys(df.columns.get_level_values(1)))
    boxes = []
    for m in names:
        values = pd.DataFrame({_level_title(lvl): df[(lvl, m)] for lvl in df.columns.get_level_values(0).unique()
                               if (lvl, m) in df.columns})
        boxes.append((m, _metric_title(m), figures.to_div(figures.level_boxplot(values, {}, _metric_title(m), False))))
    return [
        _step("a. Distributions", data=link("global_df"), html=_picker(boxes),
              desc="One value per participant for the whole graph, one box per level; hover for the ID."),
    ]


def _level_title(level: str) -> str:
    return {"node": "node", "network": "network", "network_hemi": "network, hemisphere split"}[level]


def _level_desc(params: dict, level: str) -> str:
    first = params["levels"].get(level, {})
    if any(isinstance(v, list) for v in first.get("graph_params", {}).values()):
        return f"Values are the {params['options']['summary'].upper()} over the parameter range, for each participant."
    return ""


def _method_text(level: dict) -> str:
    params = ", ".join(f"{k} {', '.join(map(str, v)) if isinstance(v, list) else v}" for k, v in level["graph_params"].items())
    return level["graph_method"] + (f" ({params})" if params else "")


def _column(nodes: pd.DataFrame | None, label_col: str, col: str) -> dict | None:
    if nodes is None or col not in nodes or label_col not in nodes:
        return None
    # XCP-D atlas tables load the label "None" (no network) as missing
    values = ["None" if pd.isna(v) else str(v) for v in nodes[col]]
    return dict(zip(nodes[label_col], values))


def _palette(networks: dict | None) -> dict:
    if not networks:
        return {}
    import matplotlib.colors as mcolors

    from conngraph.viz.colormap import labels_to_colors

    names = list(networks.values())
    return {n: mcolors.to_hex(c) for n, c in zip(names, labels_to_colors(names))}


def _coordinates(nodes: pd.DataFrame | None, label_col: str, labels: list[str]) -> np.ndarray | None:
    if nodes is None or not {"x", "y", "z"} <= set(nodes.columns) or label_col not in nodes:
        return None
    return nodes.set_index(label_col)[["x", "y", "z"]].reindex(labels).to_numpy(float)


def _input_row(loaded: dict) -> list[tuple[str, str]]:
    details = ", ".join(([mne_details(loaded)] if loaded.get("measure_name") else [])
                        + [f"{k} {loaded[k]}" for k in ("atlas", "space", "chromophore", "task", "session") if loaded.get(k)]
                        + ([loaded["connectivity"]] if loaded.get("connectivity") else [])
                        + (["Ledoit-Wolf shrinkage"] if loaded.get("shrinkage") else [])
                        + (["Fisher z input"] if loaded.get("values") == "z" else [])
                        + ([f"runs concatenated for {len(loaded['combined_runs'])} participants"]
                           if loaded.get("combined_runs") else [])
                        + [f"{k} {' or '.join('none' if v is None else str(v) for v in vals)}"
                           for k, vals in (loaded.get("bids_filters") or {}).items() if k not in ("task", "space", "ses")])
    return [("Input", loaded["source"] + (f" ({details})" if details else ""))] if loaded.get("source") else []


def _input_notes(loaded: dict) -> list[str]:
    if not loaded.get("dropped"):
        return []
    dropped = loaded["dropped"]
    what = ("with missing or constant time series were dropped when the data" if loaded.get("connectivity") else
            f"with more than {100 * loaded['bad_node_threshold']:g}% missing values were dropped when the matrices")
    return [f"{len(dropped)} nodes {what} were loaded: {', '.join(map(str, dropped[:20]))}"
            + (" …" if len(dropped) > 20 else "") + "."]


def _input_warnings(loaded: dict) -> list[str]:
    """Participants left out while loading (no single matching file) or from the NBS groups (no matrix)."""
    out = []
    if loaded.get("skipped"):
        out.append(_participants(len(loaded["skipped"]), "had no single matching file and {} skipped")
                   + ": " + "; ".join(loaded["skipped"]) + ".")
    if loaded.get("split_runs"):
        split = loaded["split_runs"]
        how = ("upstream (XCP-D --combine-runs) or with --combine-runs and --connectivity" if loaded.get("source") == "XCP-D"
               else "before running conngraph")
        out.append(_participants(len(split), f"had several runs, each analysed on its own; combine them {how}, or pick "
                                 "one with --bids-filter-file")
                   + ": " + "; ".join(f"sub-{s} ({', '.join(r)})" for s, r in split.items()) + ".")
    if loaded.get("missing"):
        out.append(_participants(len(loaded["missing"]), "in the groups table had no matrix and {} left out")
                   + ": " + ", ".join(loaded["missing"]) + ".")
    return out


def _participants(n: int, text: str) -> str:
    return f"{n} participant{'' if n == 1 else 's'} " + text.format("was" if n == 1 else "were")


NO_VALUES = "<p>No region has a value for this metric.</p>"


def _no_coordinates(nodes) -> str:
    return ("Brain figures were skipped: no node table was given." if nodes is None
            else "Brain figures were skipped: the node table has no x, y, z coordinates.")


def _surfaces(surfaces: tuple[str, str] | None) -> tuple[str, str] | None:
    if surfaces is not None:
        return surfaces
    try:
        from conngraph.viz.surface import get_fsLR_surface
        return get_fsLR_surface()
    except Exception:
        return None


def _wide_to_matrix(wide: pd.DataFrame) -> pd.DataFrame:
    mean = wide.mean(axis=0)
    names: list[str] = []
    for col in wide.columns:
        for part in col.split("__"):
            if part not in names:
                names.append(part)
    M = pd.DataFrame(np.nan, index=names, columns=names)
    for col, v in mean.items():
        a, b = col.split("__")
        M.loc[a, b] = M.loc[b, a] = v
    return M
