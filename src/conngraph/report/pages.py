"""
Assemble and write the graph-metrics and NBS reports.
"""

from __future__ import annotations

import base64
import html
import io
import json
import os
import warnings
from functools import cache
from importlib.resources import files

import numpy as np
import pandas as pd

from conngraph.report import figures
from conngraph.report.methods import graph_methods, nbs_methods, participant_count

PLOTLY_CDN = "https://cdn.plot.ly/plotly-3.5.0.min.js"
_METRIC_TITLES = {"clust_coeff": "Clustering coefficient", "btwn_cent": "Betweenness centrality",
                  "strength": "Strength", "ge_local": "Local efficiency"}
_SWEEP_LABELS = {"density": "Density", "threshold": "Threshold", "alpha": "Significance level"}
_MAX_BRAIN_EDGES = 300


def save_graph_report(result, path, nodes: pd.DataFrame | None = None, surfaces: tuple[str, str] | None = None,
                      static_brain: bool = True) -> None:
    """Write the HTML report of a compute_graph_metrics result; see GraphMetricsResult.save_report."""
    params = result.params
    opts, levels = params["options"], params["levels"]
    nodes = result.nodes if nodes is None else nodes
    label_col, network_col = opts["label_col"], opts["network_col"]
    networks = _column(nodes, label_col, network_col)
    palette = _palette(networks)
    metrics = _metric_names(result)
    notes: list[str] = [f"{n} ROIs labelled “{k}” were left out of the network levels; the node level keeps them."
                        for k, n in params.get("excluded_rois", {}).items()]

    body, sections = [], []
    for level, df, corr, sid, title in (("network", result.network_df, result.net_corr_df, "Network", "Network level"),
                                        ("network_hemi", result.net_hemi_df, result.net_hemi_corr_df, "NetworkHemi",
                                         "Network level (hemispheres)")):
        if df is None:
            continue
        hemi = level == "network_hemi"
        net_of = (lambda n: n.partition("_")[2]) if hemi else (lambda n: n)
        boxes = [(m, _metric_title(m), figures.to_div(figures.level_boxplot(df[m], palette, _metric_title(m), hemi)))
                 for m in metrics]
        M = _wide_to_matrix(corr)
        zmax = float(np.nanmax(np.abs(M.values))) or 1.0
        heat = figures.ordered_heatmap(M.values, list(M.index), [net_of(n) for n in M.index], zmax,
                                       "Fisher z" if opts.get("apply_fisher_z", True) else "r")
        means = pd.DataFrame({_metric_title(m): df[m].mean() for m in metrics})
        means.insert(0, "Node", means.index)
        body.append(dict(id=sid, title=title, desc=_level_desc(params, level), steps=[
            _step("a. Metric distributions", picker=True, html=_picker(f"{sid}-box", boxes),
                  desc="One box per network node" + (", left hemisphere darker than right" if hemi else "")
                       + "; each point is a participant (hover for the ID)."),
            _step("b. Connectivity", html=figures.to_div(heat),
                  desc="Group mean connectivity within (diagonal) and between network nodes: the matrix these graphs were built from."),
            _step("c. Group means", open=False, html=f'<div class="scroll">{_table(means.reset_index(drop=True))}</div>',
                  desc="Mean over participants of each network node's value."),
        ]))
        sections.append((sid, title))

    if result.node_df is not None:
        body.append(dict(id="Node", title="Node level", desc=_level_desc(params, "node"),
                         steps=_node_steps(result, metrics, nodes, label_col, networks, palette, surfaces,
                                           static_brain, notes)))
        sections.append(("Node", "Node level"))

    if result.curves is not None:
        first = levels.get("node") or next(iter(levels.values()))
        sweep = next(k for k, v in first["graph_params"].items() if isinstance(v, list))
        fig = figures.curves_figure(result.curves, metrics, {m: _metric_title(m) for m in metrics},
                                    _SWEEP_LABELS.get(sweep, sweep))
        body.append(dict(id="Curves", title="Across " + _SWEEP_LABELS.get(sweep, sweep).lower() + " values", desc="", steps=[
            _step("a. Metrics at each value", html=figures.to_div(fig),
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
                    static_brain: bool = True, label_col: str = "label", network_col: str = "network") -> None:
    """Write the HTML report of a run_nbs result; see NBSResult.save_report."""
    params, o = result.params, result.params["options"]
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
        _step("a. Components above threshold", html=_table(comp, sortable=False),
              desc="Connected components of suprathreshold edges, by p-value."),
        _step("b. Null distribution", html=figures.to_div(hist),
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
        if xyz is not None:
            degree = adj_sig.sum(axis=1)
            meshes = figures.surface_meshes(_surfaces(surfaces))
            parts = []
            if static_brain:
                parts.append('<div class="option-label">Option 1: static</div>'
                             + _img(_static_nbs_brain(nodes, label_col, network_col, labels, drawn, diff, degree,
                                                      palette, surfaces)))
            parts.append('<div class="option-label">Option 2: interactive</div>' + figures.to_div(figures.brain_edges(
                xyz, labels, nets or ["None"] * len(labels), drawn, np.array([diff[i, j] for i, j in drawn]), degree,
                palette, meshes, ("Group 1 > Group 2", "Group 1 < Group 2"))))
            steps.append(_step(f"{next(letter)}. On the brain", hint="(static and interactive)", html="".join(parts),
                               desc=f"Showing {what}, coloured by the sign of the group difference (group 1 − group 2); "
                                    "node size is the number of significant edges at each region."))
        else:
            notes.append(_no_coordinates(nodes))
        if nets:
            import networkx as nx

            G = nx.Graph()
            G.add_nodes_from(range(len(labels)))
            G.add_weighted_edges_from((int(i), int(j), diff[i, j]) for i, j in iu)
            steps.append(_step(f"{next(letter)}. On a circle", html=_circos_images(G, labels, nets, palette, f"{g1} − {g2}"),
                               desc=f"All {n_sig} significant edges with regions grouped by network: coloured by the "
                                    "group difference, then bundled through their networks and coloured by the "
                                    "networks they join."))
        marks = [(labels[i], labels[j]) for i, j in iu] + [(labels[j], labels[i]) for i, j in iu]
        steps.append(_step(f"{next(letter)}. Group difference", html=figures.to_div(figures.ordered_heatmap(
            diff, labels, nets, float(np.abs(diff).max()) or 1.0, "G1 − G2", marks)),
            desc="Group 1 minus group 2 for every edge" + (", ordered by network" if nets else "")
                 + "; significant edges keep their colour and the others are faded."))
        steps.append(_step(f"{next(letter)}. Group means and difference",
                           html=_img(_nbs_matrices(result, adj_sig, labels, nets, palette, (g1, g2))),
                           desc="Mean connectivity of each group and their difference, with the significant edges "
                                "shown at full strength."))
        rows = []
        for i, j in iu:
            row = {"ROI A": labels[i], "ROI B": labels[j]}
            if nets:
                row |= {"Network A": nets[i], "Network B": nets[j]}
            rows.append(row | {"Group 1": result.mean_g1[i, j], "Group 2": result.mean_g2[i, j],
                               "Difference": diff[i, j], "Component": int(result.adj[i, j])})
        edges = pd.DataFrame(rows).sort_values("Difference", key=np.abs, ascending=False)
        if nets:
            names = sorted(set(nets))
            names = [names[i] for i in figures.network_order(names, names)]
            counts = pd.DataFrame(0, index=names, columns=names)
            for i, j in iu:
                counts.loc[nets[i], nets[j]] += 1
                if nets[i] != nets[j]:
                    counts.loc[nets[j], nets[i]] += 1
            keep = [n for n in names if counts.loc[n].sum() > 0]
            steps.append(_step(f"{next(letter)}. By network pair", html=figures.to_div(figures.count_heatmap(counts.loc[keep, keep])),
                               desc="Number of significant edges within and between networks."))
        steps.append(_step(f"{next(letter)}. Edge list", open=False,
                           html=f'<div class="scroll">{_table(edges)}</div>',
                           desc=f"All {n_sig} significant edges, largest difference first; click a column to sort."))
        body.append(dict(id="Edges", title="Significant edges", desc="", steps=steps))
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
         ("ConnGraph", "v" + params["packages"]["conngraph"]),
         ("Components", len(result.pval)),
         ("Significant (p < 0.05)", _flag(len(sig), bool(sig), "")),
         ("Smallest p", _fmt_p(min(result.pval), k) if len(result.pval) else "–")],
    ]
    _write(path, "Network-based statistic report", params, sections, summary, ("run_nbs", ["matrices_g1", "matrices_g2"], o),
           body, _input_warnings(loaded), notes, nbs_methods(params), f"{len(params['groups']['g1'])} vs {len(params['groups']['g2'])}")


def _node_steps(result, metrics, nodes, label_col, networks, palette, surfaces, static_brain, notes) -> list[dict]:
    steps = []
    labels = list(result.node_df.columns.get_level_values(1).unique())
    xyz = _coordinates(nodes, label_col, labels)
    has_xyz = xyz is not None
    if has_xyz:
        keep = ~np.isnan(xyz).any(axis=1)
        meshes = figures.surface_meshes(_surfaces(surfaces))
        nets = [networks.get(lab, "None") if networks is not None else "" for lab in labels]
        static, interactive = [], []
        for m in metrics:
            values = result.node_df[m].mean(axis=0).reindex(labels).to_numpy(float)
            t = _metric_title(m)
            if static_brain:
                static.append((m, t, _img(_static_node_brain(result, nodes, label_col, labels, values, t, surfaces))))
            shown = keep & ~np.isnan(values)
            interactive.append((m, t, figures.to_div(figures.brain_values(
                xyz[shown], [lab for lab, k in zip(labels, shown) if k], [n for n, k in zip(nets, shown) if k],
                values[shown], t, meshes))))
        parts = (['<div class="option-label">Option 1: static</div>' + _picker("node-static", static)] if static else [])
        parts.append('<div class="option-label">Option 2: interactive</div>' + _picker("node-3d", interactive))
        steps.append(_step("a. Group mean on the brain", picker=True, hint="(static and interactive)", html="".join(parts),
                           desc="Colour and size both show the mean over participants. The interactive view can be "
                                "rotated and hovered for region names, on a simplified surface."))
    else:
        notes.append(_no_coordinates(nodes))
    letter = iter("abcdef"[1 if has_xyz else 0:])
    if networks is not None:
        boxes = []
        for m in metrics:
            values = result.node_df[m].mean(axis=0)
            nets = pd.Series([networks.get(lab, "None") for lab in values.index], index=values.index)
            boxes.append((m, _metric_title(m), figures.to_div(figures.node_boxplot(values, nets, palette, _metric_title(m)))))
        steps.append(_step(f"{next(letter)}. Values by network", picker=True, html=_picker("node-box", boxes),
                           desc="Mean over participants of each region; hover for its name."))
    tops = []
    for m in metrics:
        top = result.node_df[m].mean(axis=0).sort_values(ascending=False).head(15)
        table = pd.DataFrame({"ROI": top.index})
        if networks is not None:
            table["Network"] = [networks.get(lab, "None") for lab in top.index]
        table["Group mean"] = top.values
        tops.append((m, _metric_title(m), _table(table)))
    steps.append(_step(f"{next(letter)}. Highest regions", picker=True, open=False, html=_picker("node-top", tops),
                       desc="The 15 regions with the highest mean over participants."))
    if result.mean_matrix is not None:
        M = result.mean_matrix
        names = list(M.index)
        groups = [networks.get(n, "None") for n in names] if networks is not None else None
        steps.append(_step(f"{next(letter)}. Connectivity", open=False, html=figures.to_div(figures.ordered_heatmap(
            M.to_numpy(float), names, groups, 1.0, "r")),
            desc="Group mean connectivity between all regions" + (", ordered by network" if groups else "") + "."))
    group = _group_graph(result) if networks is not None else None
    if group is not None:
        import conngraph as bnv

        G, names, how = group
        nets = [networks.get(n, "None") for n in names]
        circos = _circos_images(G, names, nets, palette, "group mean r")
        spring, _ = bnv.spring_plot(G, names, nets, net2color=palette, figsize=(11, 11), network_hulls=False)
        steps.append(_step(f"{next(letter)}. Group network", html=circos + _figure_block("Spring layout", _png(spring)),
                           desc=f"The group mean connectivity turned into a graph the way each participant's was "
                                f"({how}); the metrics above come from each participant's own graph. The circle "
                                "groups regions by network; the bundled version routes edges through their networks "
                                "and colours them by the networks they join."))
    return steps


def _group_graph(result):
    """The group mean matrix built into a graph with the node level's method, or None when it cannot be redone."""
    import networkx as nx

    from conngraph.graph_theory.sparsify import GRAPH_METHODS, apply_sign, build_adjacency

    level = result.params["levels"].get("node") or {}
    method = level.get("graph_method")
    # a custom function cannot be rebuilt from its name, and PMFG takes minutes on a full atlas
    if result.mean_matrix is None or method not in GRAPH_METHODS or method == "pmfg":
        return None
    params, how = dict(level.get("graph_params") or {}), _method_text(level)
    for key, value in params.items():
        if isinstance(value, list):
            params[key] = value[len(value) // 2]
            how = f"{level['graph_method']}, {key} {params[key]} from the middle of the range"
    opts = result.params["options"]
    M = np.nan_to_num(result.mean_matrix.to_numpy(float))
    if opts.get("apply_fisher_z", True):
        M = np.tanh(M)
    np.fill_diagonal(M, 0)
    M = apply_sign(M, level["sign"])
    if opts.get("normalize_weights") and np.abs(M).max() > 0:
        M = M / np.abs(M).max()
    A = build_adjacency(M, method, params, level["sign"] == "signed")
    return nx.from_numpy_array(A), list(result.mean_matrix.index), f"{how}, {level['sign']} weights"


def _circos_images(G, labels, nets, palette, colorbar_title) -> str:
    import conngraph as bnv

    (curved, _), (bundled, _) = bnv.circos_plot(G, labels, nets, net2color=palette, figsize=(11, 11),
                                                label_fontsize=3.5, edge_colorbar_title=colorbar_title)
    return (_figure_block("Circos", _png(curved))
            + _figure_block("Circos, bundled through networks and coloured by network", _png(bundled)))


def _nbs_matrices(result, adj_sig, labels, nets, palette, group_names) -> str:
    import conngraph as bnv

    off = ~np.eye(len(labels), dtype=bool)
    lim = float(max(np.abs(result.mean_g1[off]).max(), np.abs(result.mean_g2[off]).max())) or 1.0
    layout = dict(network_labels=nets, network_palette=palette) if nets else {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig = bnv.plot_nbs_matrices(result.mean_g1, result.mean_g2, adj_sig.astype(float), labels, vmin=-lim, vmax=lim,
                                    group_names=group_names, **layout)
    return _png(fig)


def _figure_block(label: str, b64: str) -> str:
    return f'<div class="option-label">{html.escape(label)}</div>' + _img(b64)


def _static_node_brain(result, nodes, label_col, labels, values, title, surfaces) -> str:
    import conngraph as bnv

    nd = nodes.rename(columns={label_col: "label"}).copy()
    nd[title] = nd["label"].map(dict(zip(labels, values)))
    nd = nd[nd["label"].isin(labels)]
    M = result.mean_matrix if result.mean_matrix is not None else pd.DataFrame(0.0, index=labels, columns=labels)
    left, right = _surfaces(surfaces) or (None, None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds = bnv.load(M, nd)
        fig = bnv.BrainNetPlotter(ds).plot_views(
            views=[{"view": "L"}, {"view": "S"}, {"view": "R"}], node_color=title, node_size=title, node_cmap="viridis",
            node_size_range=(1.5, 6.0), edge_threshold=2.0, surface_L=left, surface_R=right, surface_alpha=0.12,
            legend=["node_color"], width=10.5, panel_size=500)
    return _png(fig)


def _static_nbs_brain(nodes, label_col, network_col, labels, drawn, diff, degree, palette, surfaces) -> str:
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
            node_palette=palette or None, width=10.5, panel_size=500)
    return _png(fig)


def _write(path, title, params, sections, summary, call, body, errors, notes, methods, chip) -> None:
    command = params.get("command")
    fn, args, opts = call
    call_text = command or f"{fn}(\n    " + ",\n    ".join(args + [f"{k}={json.dumps(v)}" for k, v in opts.items()]) + ",\n)"
    page = _environment().get_template("report.html.j2").render(
        page_title=title, heading=title, version=params["packages"]["conngraph"],
        created=params["created"].replace("T", " "), chip=chip, sections=sections, summary=summary,
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


def _step(title: str, html: str, desc: str = "", open: bool = True, picker: bool = False, hint: str = "") -> dict:
    return dict(title=title, html=html, desc=desc, open=open, picker=picker, hint=hint)


def _picker(group: str, panes: list[tuple[str, str, str]]) -> str:
    """A select showing one pane at a time; panes are (key, label, html)."""
    if len(panes) == 1:
        return panes[0][2]
    options = "".join(f'<option value="{k}">{html.escape(t)}</option>' for k, t, _ in panes)
    bodies = "".join(f'<div class="pane{" on" if i == 0 else ""}" data-group="{group}" data-key="{k}">{h}</div>'
                     for i, (k, _, h) in enumerate(panes))
    return f'<label class="hint">Metric </label><select class="picker" data-group="{group}">{options}</select>{bodies}'


def _table(df: pd.DataFrame, sortable: bool = True) -> str:
    head = "".join(f'<th class="{"num" if pd.api.types.is_numeric_dtype(df[c]) else ""}">{html.escape(str(c))}</th>'
                   for c in df.columns)
    rows = []
    for row in df.itertuples(index=False):
        cells = []
        for v in row:
            if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float, np.integer, np.floating)):
                cells.append(f"<td>{v}</td>")
            elif isinstance(v, (int, np.integer)):
                cells.append(f'<td class="num" data-v="{v}">{v}</td>')
            else:
                cells.append(f'<td class="num" data-v="{v:.10g}">{v:.4g}</td>')
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f'<table class="flat{" sortable" if sortable else ""}"><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table>'


def _png(fig) -> str:
    import matplotlib.pyplot as plt

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _img(b64: str) -> str:
    return f'<img class="figure" src="data:image/png;base64,{b64}" alt="">'


def _flag(value, ok: bool, bad_class: str) -> str:
    return f'<span class="{"ok" if ok else bad_class}">{value}</span>'


def _fmt_p(p: float, k: int) -> str:
    return f"< {1 / k:.3g}" if p == 0 else f"{p:.3g}"


def _p_relation(p: float, k: int) -> str:
    return _fmt_p(p, k) if p == 0 else f"= {_fmt_p(p, k)}"


def _metric_names(result) -> list[str]:
    for df in (result.node_df, result.net_hemi_df, result.network_df):
        if df is not None:
            return list(df.columns.get_level_values(0).unique())
    return []


def _metric_title(name: str) -> str:
    metric, _, variant = name.partition(".")
    base = _METRIC_TITLES.get(metric, metric)
    if variant.endswith(".norm"):
        return f"{base} ({variant[:-5]}, normalized)"
    return f"{base} ({variant})" if variant else base


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
    details = ", ".join([f"{k} {loaded[k]}" for k in ("atlas", "space", "chromophore", "task", "session") if loaded.get(k)]
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
