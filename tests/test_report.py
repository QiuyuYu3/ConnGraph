import json
import re

import numpy as np
import pandas as pd
import pytest

from brainnet3d.graph_theory import compute_graph_metrics
from brainnet3d.graph_theory.nbs import run_nbs
from brainnet3d.report.methods import graph_methods, nbs_methods
from brainnet3d.report.pages import _column

pytest.importorskip("bct")


@pytest.fixture(scope="module")
def graph_result(dataset):
    # Three toy networks are too few for TMFG at the whole-network level
    return compute_graph_metrics(
        dataset.matrices, dataset.nodes_df, level="both", hemi_split="both", network_col="network",
        network_graph_method="full", n_jobs=1, verbose=False,
    )


@pytest.fixture(scope="module")
def nbs_result(groups):
    g1, g2 = groups
    return run_nbs(g1, g2, thresh=1.0, k=20, seed=0, verbose=False)


def test_graph_result_keeps_nodes_and_fisher_mean_matrix(dataset, graph_result):
    mats = [m.to_numpy(float) for m in dataset.matrices.values()]
    z = np.mean([np.arctanh(np.clip(m, -1 + 1e-7, 1 - 1e-7)) for m in mats], axis=0)
    np.fill_diagonal(z, 0)
    expected = np.tanh(z)
    np.testing.assert_allclose(graph_result.mean_matrix.to_numpy(), expected, atol=1e-12)
    assert list(graph_result.mean_matrix.index) == list(dataset.matrices[next(iter(dataset.matrices))].index)
    assert graph_result.nodes.equals(dataset.nodes_df)


def test_nbs_result_keeps_group_means(groups, nbs_result):
    g1, g2 = groups
    np.testing.assert_allclose(nbs_result.mean_g1, np.mean([m.to_numpy(float) for m in g1.values()], axis=0))
    np.testing.assert_allclose(nbs_result.mean_g2, np.mean([m.to_numpy(float) for m in g2.values()], axis=0))


def test_graph_methods_follow_the_users_text_for_the_default_run(graph_result):
    text = graph_methods(graph_result.params)
    plain = text["plain"]
    assert "Matrices were loaded for 6 participants." in plain
    assert "Triangulated Maximally Filtered Graph (TMFG) algorithm [TODO: reference for the TMFG method]" in plain
    assert "Edges were selected by the absolute value of their weights and kept their original signs." in plain
    assert "(van den Heuvel et al., 2017)" in plain and "(Jiang et al., 2023)" in plain
    assert "following the Brain Connectivity Toolbox (BCT; Rubinov & Sporns, 2010)" in plain
    assert "signed-weight generalization (Costantini & Perugini, 2014)" in plain
    assert "The remaining three metrics were computed on the absolute values of the edge weights." in plain
    assert "Analyses were conducted at three levels." in plain
    assert "At the hemisphere-separated network level, left- and right-hemisphere ROIs" in plain
    assert "Network-level graphs were constructed with the full method instead." in plain
    refs = plain.split("References")[1].strip().splitlines()
    assert [r.split(",")[0] for r in refs] == ["Costantini", "van den Heuvel", "Jiang", "Rubinov"]
    assert "\\cite{vandenHeuvel2017}" in text["latex"] and "TODO" in text["latex"]
    assert 'class="todo"' in text["html"]


def test_graph_methods_describe_an_xcpd_input(dataset):
    atlas = dataset.nodes_df.copy()
    atlas.attrs["brainnet3d_input"] = {
        "source": "XCP-D", "atlas": "Gordon", "space": "fsLR", "task": "rest", "session": "ses-01", "n_loaded": 6,
        "bad_node_threshold": 0.9, "drop_mode": "union", "dropped": [],
    }
    result = compute_graph_metrics(dataset.matrices, atlas, level="node", network_col="network", metrics=["strength"],
                                   n_jobs=1, verbose=False)
    assert result.params["input"]["source"] == "XCP-D"
    plain = graph_methods(result.params)["plain"]
    assert ("Functional connectivity matrices derived from XCP-D (Gordon atlas; fsLR space; Pearson's r) were used "
            "as input for graph theory analysis") in plain
    assert ("Nodes for which more than 90% of connectivity values were missing in any participant were excluded "
            "across all participants.") in plain
    assert "each of the 24 parcels of the Gordon atlas." in plain


def _methods_with_input(dataset, record):
    atlas = dataset.nodes_df.copy()
    atlas.attrs["brainnet3d_input"] = {"n_loaded": 6, "bad_node_threshold": 0.9, "drop_mode": "union",
                                       "dropped": [], **record}
    result = compute_graph_metrics(dataset.matrices, atlas, level="node", network_col="network", metrics=["strength"],
                                   n_jobs=1, verbose=False)
    return result.params, graph_methods(result.params)["plain"]


def test_graph_methods_describe_xcpd_time_series(dataset):
    params, plain = _methods_with_input(dataset, {
        "source": "XCP-D", "atlas": "Gordon", "space": "fsLR", "connectivity": "partial correlation", "shrinkage": True,
    })
    assert params["packages"]["nilearn"]
    assert ("Functional connectivity matrices computed from XCP-D regional time series (Gordon atlas; fsLR space) "
            "were used as input") in plain
    assert "Connectivity was estimated as partial correlation" in plain
    assert "Ledoit-Wolf shrinkage" in plain and f"nilearn v{params['packages']['nilearn']}" in plain
    assert "Nodes whose time series had missing or constant values in any participant were excluded" in plain
    assert "Pearson's r" not in plain and "of connectivity values were missing" not in plain


def test_graph_methods_describe_other_time_series(dataset):
    _, plain = _methods_with_input(dataset, {"source": "time series", "connectivity": "correlation", "shrinkage": False})
    assert "Functional connectivity matrices computed from regional time series were used as input" in plain
    assert "Connectivity was estimated as Pearson correlation with nilearn" in plain
    assert "Ledoit-Wolf" not in plain and "XCP-D" not in plain


def test_graph_methods_note_fisher_z_input(dataset):
    _, plain = _methods_with_input(dataset, {"source": "matrix files", "values": "z"})
    assert "The input matrices contained Fisher z values, which were converted back to correlation coefficients." in plain
    _, plain = _methods_with_input(dataset, {"source": "matrix files", "values": "r"})
    assert "Fisher z values" not in plain


def test_nbs_methods_describe_time_series_input(nbs_result):
    params = {**nbs_result.params, "input": {"source": "time series", "connectivity": "correlation", "shrinkage": False}}
    assert "Connectivity was estimated as Pearson correlation" in nbs_methods(params)["plain"]


def test_graph_methods_from_load_group_state_the_node_rule(graph_result):
    assert graph_result.params["input"]["source"] == "matrix files"
    plain = graph_methods(graph_result.params)["plain"]
    assert "matrices were used as input" in plain and "XCP-D" not in plain
    assert "more than 90% of connectivity values were missing in any participant" in plain


def test_graph_methods_describe_a_hemisphere_level_alone(dataset):
    result = compute_graph_metrics(dataset.matrices, dataset.nodes_df, level="network", network_col="network",
                                   metrics=["strength"], verbose=False)
    plain = graph_methods(result.params)["plain"]
    assert "Analyses were conducted at one level." in plain
    assert "averaged within and between the left- and right-hemisphere ROIs of each network" in plain
    assert "The remaining" not in plain


def test_graph_methods_describe_a_density_range(dataset):
    result = compute_graph_metrics(dataset.matrices, dataset.nodes_df, level="node", network_col="network",
                                   metrics=["strength"], graph_method="density",
                                   graph_params={"density": [0.2, 0.3]}, n_jobs=1, verbose=False)
    plain = graph_methods(result.params)["plain"]
    assert "Negative weights were set to zero" in plain
    assert "0.20 and 0.30" in plain
    assert "area under the curve" in plain


def test_nbs_methods_name_the_test_and_settings(nbs_result):
    plain = nbs_methods(nbs_result.params)["plain"]
    assert "network-based statistic [TODO: reference for the network-based statistic]" in plain
    assert "two-sample t-test" in plain and "20 permutations" in plain and "exceeded 1.0" in plain


def _sections(text):
    return re.findall(r'<h2 id="([^"]+)"', text)


def test_graph_report_has_every_section_and_the_methods(graph_result, surfaces, tmp_path):
    path = tmp_path / "graph.html"
    graph_result.save_report(path, surfaces=surfaces, static_brain=False)
    text = path.read_text(encoding="utf-8")
    assert _sections(text) == ["Summary", "Network", "NetworkHemi", "Node", "Errors", "Methods", "Versions"]
    assert "https://cdn.plot.ly/" in text
    assert "Triangulated Maximally Filtered Graph" in text
    assert "Option 2: interactive" in text and "Option 1: static" not in text
    assert text.count('class="plotly-graph-div"') >= 8


def test_graph_report_renders_the_static_brain(graph_result, surfaces, tmp_path):
    path = tmp_path / "graph.html"
    graph_result.save_report(path, surfaces=surfaces, static_brain=True)
    text = path.read_text(encoding="utf-8")
    assert "Option 1: static" in text and "data:image/png;base64," in text


def test_graph_report_without_coordinates_skips_the_brain(dataset, tmp_path):
    atlas = dataset.nodes_df.drop(columns=["x", "y", "z"])
    result = compute_graph_metrics(dataset.matrices, atlas, level="node", network_col="network",
                                   metrics=["strength"], n_jobs=1, verbose=False)
    path = tmp_path / "graph.html"
    result.save_report(path)
    text = path.read_text(encoding="utf-8")
    assert "Option 2: interactive" not in text
    assert "no x, y, z coordinates" in text


def test_graph_report_summary_names_the_time_series_measure(dataset, tmp_path):
    atlas = dataset.nodes_df.drop(columns=["x", "y", "z"])
    atlas.attrs["brainnet3d_input"] = {"source": "time series", "connectivity": "partial correlation", "shrinkage": True,
                                       "n_loaded": 6, "bad_node_threshold": 0.9, "drop_mode": "union", "dropped": ["r1"]}
    result = compute_graph_metrics(dataset.matrices, atlas, level="node", network_col="network",
                                   metrics=["strength"], n_jobs=1, verbose=False)
    result.save_report(tmp_path / "graph.html")
    text = (tmp_path / "graph.html").read_text(encoding="utf-8")
    assert "time series (partial correlation, Ledoit-Wolf shrinkage)" in text
    assert "1 nodes with missing or constant time series were dropped when the data were loaded: r1." in text


def test_nbs_report_lists_components_and_settings(nbs_result, dataset, surfaces, tmp_path):
    path = tmp_path / "nbs.html"
    nbs_result.save_report(path, nodes=dataset.nodes_df, surfaces=surfaces, static_brain=False)
    text = path.read_text(encoding="utf-8")
    assert _sections(text)[:2] == ["Summary", "Components"]
    assert "network-based statistic" in text
    assert json.dumps(nbs_result.params["options"]["thresh"]) in text


def test_reports_read_missing_network_labels_as_none(nbs_result, graph_result, dataset, tmp_path):
    # XCP-D atlas tables load the label "None" as missing
    nodes = dataset.nodes_df.copy()
    nodes["network"] = nodes["network"].astype("string")
    nodes.loc[[0, 5], "network"] = pd.NA
    networks = _column(nodes, "label", "network")
    assert networks[nodes["label"][0]] == "None" and all(isinstance(v, str) for v in networks.values())
    nbs_result.save_report(tmp_path / "nbs.html", nodes=nodes, static_brain=False)
    graph_result.save_report(tmp_path / "graph.html", nodes=nodes, static_brain=False)


def test_nbs_report_works_without_a_node_table(nbs_result, tmp_path):
    path = tmp_path / "nbs.html"
    nbs_result.save_report(path)
    assert "Components" in path.read_text(encoding="utf-8")
