"""
Participant reports: one page per participant, with a section per session and atlas or chromophore.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from conngraph.report import figures
from conngraph.report.methods import graph_methods
from conngraph.report.pages import (
    NO_VALUES,
    _SWEEP_LABELS,
    _circos_images,
    value_label,
    _column,
    _coordinates,
    _figure_block,
    _figure_row,
    _flag,
    _input_notes,
    _input_row,
    _level_title,
    _method_text,
    _metric_names,
    _metric_title,
    _no_coordinates,
    _palette,
    _picker,
    _rebuilt_graph,
    _static_node_brain,
    _step,
    _linker,
    _table,
    _wide_to_matrix,
    _write,
)


def participant_section(result, sid: str, matrix: pd.DataFrame, nodes: pd.DataFrame, params: dict, section_id: str,
                        title: str, meshes: list | None, figs=None, surfaces: tuple[str, str] | None = None) -> dict:
    """One participant's results in one session and atlas as plain data; figs saves the static brain views."""
    opts = params["options"]
    networks = _column(nodes, opts["label_col"], opts["network_col"])
    palette = _palette(networks)
    cols = list(matrix.columns)
    labels = [str(c) for c in cols]
    nets = [networks.get(c, "None") for c in cols] if networks is not None else None
    graph = _rebuilt_graph(matrix, params, is_z=False)
    letter = iter("abcdefghij")
    link = _linker(result.files, figs.page) if figs is not None else (lambda *keys: [])
    errors = [f"{lvl}: {subs[sid]}" for lvl, subs in result.failed.items() if sid in subs] + params["warnings"]
    notes = _input_notes(params.get("input") or {})

    overview = _overview(matrix, graph, params, errors).to_numpy().tolist()
    steps = []

    panes = [("regions", "Regions", figures.to_div(figures.ordered_heatmap(
        np.nan_to_num(matrix.to_numpy(float)), labels, nets, 1.0, value_label(params))))]
    for attr, name in (("net_corr_df", "Networks"), ("net_hemi_corr_df", "Networks (hemispheres)")):
        wide = getattr(result, attr)
        if wide is not None:
            M = _wide_to_matrix(wide.loc[[sid]])
            hemi = attr == "net_hemi_corr_df"
            panes.append((attr, name, figures.to_div(figures.ordered_heatmap(
                M.to_numpy(float), list(M.index), [n.partition("_")[2] if hemi else n for n in M.index],
                float(np.nanmax(np.abs(M.to_numpy(float)))) or 1.0, "Fisher z" if opts.get("apply_fisher_z") else value_label(params)))))
    steps.append(_step(f"{next(letter)}. Connectivity matrix", data=link("net_corr_df", "net_hemi_corr_df"), html=_picker(panes),
                       desc="The input matrix ordered by network, and the mean connectivity within and between networks."))

    if result.node_df is not None:
        node = result.node_df.loc[sid]
        metrics = _metric_names(result.node_df)
        xyz = _coordinates(nodes, opts["label_col"], cols)
        parts = []
        if xyz is not None and figs is not None:
            brains = []
            for m in metrics:
                values = node[m].reindex(cols).to_numpy(float)
                shown = ~np.isnan(xyz).any(axis=1) & ~np.isnan(values)
                if not shown.any():
                    brains.append((m, _metric_title(m), NO_VALUES))
                    continue
                html_ = _static_node_brain(result, nodes, opts["label_col"], cols, values, _metric_title(m), surfaces,
                                           figs, m)
                if meshes is not None:
                    html_ +=figs.save_html(figures.brain_values(
                        xyz[shown], [lab for lab, k in zip(labels, shown) if k],
                        [n for n, k in zip(nets or [""] * len(cols), shown) if k], values[shown], _metric_title(m),
                        meshes), f"brain_{m}")
                brains.append((m, _metric_title(m), html_))
            parts.append('<div class="option-label">On the brain</div>' + _picker(brains, columns=1))
        elif xyz is None:
            notes.append(_no_coordinates(nodes))
        if nets is not None:
            boxes = [(m, _metric_title(m), figures.to_div(figures.node_boxplot(
                node[m], pd.Series(nets, index=node[m].index), palette, _metric_title(m)))) for m in metrics]
            parts.append('<div class="option-label">By network</div>' + _picker(boxes, columns=1))
        if parts:
            steps.append(_step(f"{next(letter)}. Node metrics", data=link("node_df"), html="".join(parts),
                               desc="Each region's value; hover for its name."))

    tables = []
    for attr, level in (("network_df", "network"), ("net_hemi_df", "network_hemi")):
        df = getattr(result, attr)
        if df is not None:
            table = df.loc[sid].unstack(0)
            table.columns = [_metric_title(m) for m in table.columns]
            tables.append(f'<div class="option-label">{_level_title(level).capitalize()}</div>'
                          + _table(table.rename_axis("Network").reset_index()))
    if tables:
        steps.append(_step(f"{next(letter)}. Network metrics", data=link("network_df", "net_hemi_df"), html="".join(tables),
                           desc="Each network node's value in the network-level graph."))

    if result.global_df is not None:
        table = result.global_df.loc[sid].unstack(0)
        table.columns = [_level_title(c).capitalize() for c in table.columns]
        table.insert(0, "Metric", [_metric_title(m) for m in table.index])
        steps.append(_step(f"{next(letter)}. Whole graph", data=link("global_df"), html=_table(table.reset_index(drop=True), sortable=False),
                           desc="One value per graph."))

    if graph is not None and nets is not None:
        G, names, how = graph
        spring = figures.to_div(figures.spring_figure(G, names, nets, palette))
        steps.append(_step(f"{next(letter)}. Graph", html=_figure_row(*_circos_images(G, names, nets, palette, value_label(params)))
                           + _figure_block("Spring layout", spring),
                           desc=f"This participant's node-level graph ({how}). Hover a region for its name."))

    if result.curves is not None:
        curves = result.curves[result.curves["ID"] == sid]
        first = params["levels"].get("node") or next(iter(params["levels"].values()))
        sweep = next(k for k, v in first["graph_params"].items() if isinstance(v, list))
        shown = list(dict.fromkeys(curves["metric"]))
        steps.append(_step(f"{next(letter)}. Across {_SWEEP_LABELS.get(sweep, sweep).lower()} values",
                           data=link("curves"), html=figures.to_div(
            figures.curves_figure(curves, shown, {m: _metric_title(m) for m in shown}, _SWEEP_LABELS.get(sweep, sweep))),
            desc="Mean over nodes at each value: the values the summary integrates."))

    return dict(id=section_id, title=title, desc="", steps=steps, errors=errors, notes=notes, params=params,
                overview=overview)


def save_participant_report(label: str, sections: list[dict], path) -> None:
    """Write one participant's page from the sections of each session and atlas."""
    params = sections[0]["params"]
    levels = params["levels"]
    first = levels.get("node") or next(iter(levels.values()))
    errors = [f"{s['title']}: {e}" for s in sections for e in s["errors"]]
    notes = list(dict.fromkeys(n for s in sections for n in s["notes"]))
    failed = sum(1 for s in sections for e in s["errors"] if e not in s["params"]["warnings"])
    warned = sum(len(s["params"]["warnings"]) for s in sections)
    summary = [
        _input_row(params.get("input") or {})
        + [("Participant", f"sub-{label}"),
           ("Analyses", ", ".join(s["title"] for s in sections)),
           ("Levels", "; ".join(f"{_level_title(n)} ({d['n_nodes']} nodes)" for n, d in levels.items())),
           ("Graph construction", _method_text(first)),
           ("Metrics", ", ".join(dict.fromkeys(m for d in levels.values() for m in d["metrics"])))],
        [("Created", params["created"].replace("T", " ")),
         ("ConnGraph", "v" + params["packages"]["conngraph"]),
         ("Python", params["python"]),
         ("Failed", _flag(failed, failed == 0, "bad")),
         ("Warnings", _flag(warned, not warned, "warn"))],
    ]
    body = [{k: s[k] for k in ("id", "title", "desc", "steps")} for s in sections]
    # one column per session and atlas: what went into its graphs and how they turned out
    items = list(dict.fromkeys(item for s in sections for item, _ in s["overview"]))
    overview = pd.DataFrame({"": items} | {s["title"]: [dict(s["overview"]).get(i, "") for i in items] for s in sections})
    _write(path, f"sub-{label}", params, [(s["id"], s["title"]) for s in sections], summary,
           ("compute_graph_metrics", ["matrices", "atlas"], params["options"]), body, errors, notes,
           graph_methods(params), f"sub-{label}", quantities=_table(overview, sortable=False))


def _overview(matrix: pd.DataFrame, graph, params: dict, errors: list[str]) -> pd.DataFrame:
    import networkx as nx

    M = matrix.to_numpy(float)
    upper = M[np.triu_indices(len(M), 1)]
    rows = [("Nodes", str(len(M))),
            ("Missing values in the input", str(int(np.isnan(upper).sum()))),
            ("Negative connections in the input", f"{100 * np.mean(upper[~np.isnan(upper)] < 0):.1f}%")]
    if graph is not None:
        G, _, how = graph
        n = G.number_of_nodes()
        components = nx.number_connected_components(G)
        rows += [("Graph", how), ("Edges kept", f"{G.number_of_edges()} ({100 * nx.density(G):.1f}% of possible)"),
                 ("Connected pieces", _flag(components, components == 1, "warn")),
                 ("Isolated nodes", _flag(sum(1 for i in range(n) if G.degree(i) == 0), True, ""))]
    rows.append(("Problems", _flag(len(errors), not errors, "warn")))
    return pd.DataFrame(rows, columns=["Item", "Value"])
