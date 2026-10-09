"""
Methods text generated from a run record (GraphMetricsResult.params or NBSResult.params).
"""

from __future__ import annotations

import html
import re
from functools import cache
from importlib.resources import files

_FORMATS = ("plain", "markdown", "latex", "html")
_METRIC_ORDER = ("clust_coeff", "strength", "btwn_cent", "ge_local")
_SLOT = re.compile(r"\{(cite|cite_bare|todo):([^}]+)\}")
_COUNT_WORDS = {1: "one", 2: "two", 3: "three", 4: "four"}
_SWEEP_NAMES = {"density": "densities", "threshold": "thresholds", "alpha": "significance levels"}
_CHROMO = {"hbo": "HbO", "hbr": "HbR"}


@cache
def steps() -> dict:
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib
    return tomllib.loads(files("brainnet3d.report").joinpath("data/steps.toml").read_text(encoding="utf-8"))


@cache
def references() -> dict[str, dict]:
    import bibtexparser

    lib = bibtexparser.parse_string(files("brainnet3d.report").joinpath("data/references.bib").read_text(encoding="utf-8"))
    return {e.key: {f.key: f.value for f in e.fields} for e in lib.entries}


def graph_methods(params: dict) -> dict[str, str]:
    """Methods text of a compute_graph_metrics run, as {"plain", "markdown", "latex", "html"}."""
    opts, levels = params["options"], params["levels"]
    first = levels.get("node") or next(iter(levels.values()))
    loaded = params.get("input") or {}
    series = bool(loaded.get("connectivity"))
    source = ""
    if loaded.get("source") == "XCP-D":
        source = (f" computed from XCP-D regional time series ({loaded['atlas']} atlas; {loaded['space']} space)" if series
                  else f" derived from XCP-D ({loaded['atlas']} atlas; {loaded['space']} space; Pearson's r)")
    elif loaded.get("source") == "fnirs-pipe":
        source = f" derived from fnirs-pipe ({_CHROMO[loaded['chromophore']]}; Pearson's r)"
    elif series:
        source = " computed from regional time series"

    para1 = [("input", {"source": source, "ver": params["packages"]["brainnet3d"]})]
    para1 += _input_sentences(params)
    para1.append(("participants", {"n": len(params["subjects"])}))
    if loaded.get("bad_node_threshold", 1.0) < 1.0:
        scope = "any participant" if loaded.get("drop_mode", "union") == "union" else "every participant"
        para1.append(("bad_series" if series else "bad_nodes",
                      {"pct": f"{100 * loaded['bad_node_threshold']:g}", "scope": scope}))

    para2 = [_graph_sentence(first), ("sign_" + first["sign"], {})]
    if first["graph_method"] == "tmfg":
        para2.append(("why_tmfg", {}))
    if opts.get("normalize_weights"):
        para2.append(("normalize_weights", {}))
    sweep = _sweep(first)
    if sweep:
        name, values = sweep
        para2.append((f"sweep_{opts['summary']}", {"param": name, "plural": _SWEEP_NAMES.get(name, name + "s"),
                                                    "values": _join([f"{v:.2f}" for v in values])}))
    net = levels.get("network") or levels.get("network_hemi")
    if net and "node" in levels and (net["graph_method"], net["graph_params"]) != (first["graph_method"], first["graph_params"]):
        para2.append(("network_method", {"value": _method_label(net)}))
    if opts.get("n_random"):
        para2.append(("random", {"n": opts["n_random"]}))

    para3 = [("metrics_intro", {})] + _metric_sentences([m for m in first["metrics"] if not m.endswith(".norm")])

    para4 = [("levels", {"count": _count(len(levels), "level")})]
    fisher = opts.get("apply_fisher_z", True)
    z = {"z": "Fisher z-transformed connectivity" if fisher else "connectivity",
         "back": " and converted back to r" if fisher else ""}
    if "node" in levels and loaded.get("source") == "fnirs-pipe":
        para4.append(("level_node_channels", {"n": levels["node"]["n_nodes"]}))
    elif "node" in levels:
        atlas = f" of the {loaded['atlas']} atlas" if loaded.get("atlas") else ""
        para4.append(("level_node", {"n": levels["node"]["n_nodes"], "atlas": atlas}))
    if "network" in levels:
        para4.append(("level_network", {"n": levels["network"]["n_nodes"], **z}))
    if "network_hemi" in levels:
        key = "level_network_hemi" if "network" in levels else "level_network_hemi_alone"
        para4.append((key, {"n": levels["network_hemi"]["n_nodes"], **z}))
    for label, count in params.get("excluded_rois", {}).items():
        if label == "None":
            para4.append(("excluded_none", {"count": count}))
        else:
            para4.append(("excluded", {"labels": f"“{label}”", "count": count}))

    return render([para1, para2, para3, para4])


def nbs_methods(params: dict) -> dict[str, str]:
    """Methods text of a run_nbs run, as {"plain", "markdown", "latex", "html"}."""
    o = params["options"]
    slots = {
        "ver": params["packages"]["brainnet3d"],
        "test": "paired t-test" if o["paired"] else "two-sample t-test",
        "n1": len(params["groups"]["g1"]), "n2": len(params["groups"]["g2"]),
        "tail": {"both": "absolute", "right": "positive", "left": "negative"}[o["tail"]],
        "thresh": o["thresh"], "k": o["k"],
        "shuffle": "the sign of each paired difference" if o["paired"] else "the group labels",
    }
    sentences = [("nbs_intro", slots)] + _input_sentences(params)
    if (params.get("input") or {}).get("fisher_z"):
        sentences.append(("nbs_fisher", {}))
    return render([sentences + [("nbs", slots)]])


def _input_sentences(params: dict) -> list[tuple[str, dict]]:
    """How the matrices were obtained, when the loader computed them from time series or read Fisher z values."""
    loaded = params.get("input") or {}
    if loaded.get("connectivity"):
        measure = "Pearson correlation" if loaded["connectivity"] == "correlation" else loaded["connectivity"]
        key = "connectivity_shrinkage" if loaded.get("shrinkage") else "connectivity"
        combined = [("combine_runs", {"n": len(loaded["combined_runs"])})] if loaded.get("combined_runs") else []
        return combined + [(key, {"measure": measure, "ver": params["packages"].get("nilearn")})]
    if loaded.get("values") == "z":
        return [("values_z", {})]
    return []


def render(paragraphs: list[list[tuple[str, dict]]]) -> dict[str, str]:
    """Fill the step templates; paragraphs are lists of (step key, slots)."""
    refs, templates = references(), steps()
    cited: list[str] = []
    out = {}
    for fmt in _FORMATS:
        paras = []
        for para in paragraphs:
            paras.append(" ".join(_sentence(templates[key]["plain"], slots, fmt, refs, cited) for key, slots in para))
        out[fmt] = paras

    reflist = [_apa_reference(refs[k]) for k in sorted(cited, key=lambda k: refs[k].get("sortkey", refs[k]["author"]).lower())]
    return {
        "plain": "\n\n".join(out["plain"]) + ("\n\nReferences\n\n" + "\n".join(reflist) if reflist else ""),
        "markdown": "\n\n".join(out["markdown"]) + ("\n\n**References**\n\n" + "\n".join(f"- {r}" for r in reflist) if reflist else ""),
        "latex": "\n\n".join(out["latex"]),
        "html": "".join(f'<p class="boilerplate-para">{p}</p>' for p in out["html"])
                + ('<h4 class="boilerplate-refs-title">References</h4><ol class="boilerplate-refs">'
                   + "".join(f"<li>{html.escape(r)}</li>" for r in reflist) + "</ol>" if reflist else ""),
    }


def _sentence(template: str, slots: dict, fmt: str, refs: dict, cited: list[str]) -> str:
    # Citation slots are swapped for placeholders first, so str.format never sees LaTeX braces
    held: list[tuple[str, str]] = []

    def hold(m: re.Match) -> str:
        held.append((m.group(1), m.group(2)))
        return f"\x00{len(held) - 1}\x00"

    text = _SLOT.sub(hold, template).format(**slots)
    if fmt == "latex":
        text = re.sub(r"([&%#_$])", r"\\\1", text)
    elif fmt == "html":
        text = html.escape(text, quote=False)
    for i, (kind, arg) in enumerate(held):
        text = text.replace(f"\x00{i}\x00", _citation(kind, arg, fmt, refs, cited))
    return text


def _citation(kind: str, arg: str, fmt: str, refs: dict, cited: list[str]) -> str:
    if kind == "todo":
        note = f"[TODO: reference for {arg}]"
        return {"latex": f"\\textbf{{{note}}}", "markdown": f"**{note}**",
                "html": f'<span class="todo">{html.escape(note)}</span>'}.get(fmt, note)
    keys = [k.strip() for k in arg.split(",")]
    cited.extend(k for k in keys if k not in cited)
    if fmt == "latex":
        return "\\cite{" + ",".join(keys) + "}"
    text = "; ".join(_in_text(refs[k]) for k in keys)
    if kind == "cite":
        text = f"({text})"
    return f'<span class="boilerplate-cite">{html.escape(text)}</span>' if fmt == "html" else text


def _surname(author: str) -> str:
    return author.split(",")[0].strip()


def _in_text(entry: dict) -> str:
    authors = [a.strip() for a in entry["author"].split(" and ")]
    if len(authors) == 1:
        who = _surname(authors[0])
    elif len(authors) == 2:
        who = f"{_surname(authors[0])} & {_surname(authors[1])}"
    else:
        who = f"{_surname(authors[0])} et al."
    return f"{who}, {entry['year']}"


def _apa_reference(entry: dict) -> str:
    names = []
    for author in entry["author"].split(" and "):
        surname, _, given = author.partition(",")
        names.append(f"{surname.strip()}, {given.strip()}" if given.strip() else surname.strip())
    who = names[0] if len(names) == 1 else ", ".join(names[:-1]) + ", & " + names[-1]
    title = re.sub(r"[{}]", "", entry["title"])
    source = entry["journal"] + (f", {entry['volume']}" if entry.get("volume") else "")
    source += f"({entry['number']})" if entry.get("number") else ""
    source += f", {entry['pages'].replace('--', '–')}" if entry.get("pages") else ""
    line = f"{who} ({entry['year']}). {title}. {source}."
    return line + (f" https://doi.org/{entry['doi']}" if entry.get("doi") else "")


def _sweep(level: dict) -> tuple[str, list] | None:
    for name, value in level["graph_params"].items():
        if isinstance(value, list):
            return name, value
    return None


def _graph_sentence(level: dict) -> tuple[str, dict]:
    method = level["graph_method"]
    if f"graph_{method}" not in steps():
        return "graph_callable", {"value": method}
    value = next(iter(level["graph_params"].values()), None)
    if isinstance(value, list):
        value = "a value in the range below"
    return f"graph_{method}", {"value": value}


def _method_label(level: dict) -> str:
    params = ", ".join(f"{k} = {v}" for k, v in level["graph_params"].items())
    return level["graph_method"] + (f" ({params})" if params else "")


def _metric_sentences(names: list[str]) -> list[tuple[str, dict]]:
    out: list[tuple[str, dict]] = []
    template = steps()
    for metric in _METRIC_ORDER:
        variants = [n for n in names if n.split(".")[0] == metric]
        if not variants:
            continue
        if f"def_{metric}" in template:
            out.append((f"def_{metric}", {}))
        out += [(v, {}) for v in variants if v in template]
        others = {n.split(".")[0] for n in names} - {"clust_coeff"}
        if metric == "clust_coeff" and variants == ["clust_coeff.costantini"] and others:
            out.append(("remaining_abs", {"count": _count(len(others), "metric")}))
    return out


def _count(n: int, noun: str) -> str:
    return f"{_COUNT_WORDS.get(n, n)} {noun}{'' if n == 1 else 's'}"


def _join(values: list[str]) -> str:
    return values[0] if len(values) == 1 else ", ".join(values[:-1]) + " and " + values[-1]
