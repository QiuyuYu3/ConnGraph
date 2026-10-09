import json

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from conngraph.graph_theory import compute_graph_metrics
from conngraph.graph_theory.compare import compare_groups, permuted_t_test

pytest.importorskip("bct")


def _family(n=20, k=8, seed=0):
    rng = np.random.default_rng(seed)
    ids = [f"s{i:02d}" for i in range(n)]
    values = pd.DataFrame(rng.standard_normal((n, k)), index=ids, columns=[f"c{j}" for j in range(k)])
    group1 = pd.Series(np.arange(n) < n // 2, index=ids)
    values.loc[group1, "c0"] += 2.0
    return values, group1


def test_t_and_p_match_a_two_sample_t_test():
    values, group1 = _family()
    table = permuted_t_test(values, group1, n_perms=50, seed=0)
    t, p = stats.ttest_ind(values[group1], values[~group1])
    assert np.allclose(table["t"], t) and np.allclose(table["p"], p)
    assert table.loc["c0", "t"] > 0
    assert np.allclose(table["p_fdr"], stats.false_discovery_control(p))
    assert list(table["n_group1"].unique()) == [10] and np.allclose(table["mean_group1"], values[group1].mean())


def test_covariates_enter_the_model_as_in_least_squares():
    values, group1 = _family()
    cov = pd.DataFrame({"age": np.random.default_rng(1).normal(size=len(values))}, index=values.index)
    table = permuted_t_test(values, group1, cov, n_perms=50, seed=0)
    X = np.column_stack([group1.to_numpy(float), np.ones(len(values)), cov["age"]])
    beta, *_ = np.linalg.lstsq(X, values.to_numpy(), rcond=None)
    dof = len(values) - 3
    res = values.to_numpy() - X @ beta
    se = np.sqrt((res ** 2).sum(axis=0) / dof * np.linalg.inv(X.T @ X)[0, 0])
    assert np.allclose(table["t"], beta[0] / se)
    assert np.allclose(table["p"], 2 * stats.t.sf(np.abs(beta[0] / se), dof))


def test_family_wise_p_values_come_from_the_largest_t_and_do_not_depend_on_the_order_of_calls():
    values, group1 = _family()
    a = permuted_t_test(values, group1, n_perms=200, seed=3)
    b = permuted_t_test(values, group1, n_perms=200, seed=3)
    assert a.equals(b)
    assert a["p_fwe"].min() == pytest.approx(1 / 201)
    order = np.argsort(-a["t"].abs().to_numpy())
    assert np.all(np.diff(a["p_fwe"].to_numpy()[order]) >= 0)


def test_columns_with_missing_or_constant_values_are_not_tested():
    values, group1 = _family()
    values["c1"] = 1.0
    values.iloc[0, 2] = np.nan
    table = permuted_t_test(values, group1, n_perms=50, seed=0)
    assert table.loc[["c1", "c2"], ["t", "p", "p_fdr", "p_fwe"]].isna().all().all()
    tested = table["p"].notna()
    assert tested.sum() == 6 and np.allclose(table.loc[tested, "p_fdr"],
                                             stats.false_discovery_control(table.loc[tested, "p"]))


@pytest.fixture(scope="module")
def graph_result(dataset):
    return compute_graph_metrics(dataset.matrices, dataset.nodes_df, level="both", hemi_split="both",
                                 metrics=["strength", "eff_global"],
                                 graph_method="density", graph_params={"density": 0.3}, network_col="network",
                                 n_jobs=1, verbose=False)


def _labels(dataset):
    ids = sorted(dataset.matrices)
    return {sid: "A" if i < len(ids) // 2 else "B" for i, sid in enumerate(ids)}


def test_compare_groups_tests_metrics_blocks_and_edges(dataset, graph_result):
    result = compare_groups(_labels(dataset), ("A", "B"), graph_result, dataset.matrices,
                            compare=("metrics", "blocks", "edges"), n_perms=50, seed=0, verbose=False)
    tables = result.tables
    assert set(tables) == {"metrics_node", "metrics_network", "metrics_networkhemi", "global_node", "global_network",
                           "global_networkhemi", "blocks_network", "blocks_networkhemi", "edges"}
    node = tables["metrics_node"]
    assert list(node.columns[:2]) == ["metric", "node"] and len(node) == len(dataset.nodes_df)
    assert list(tables["global_node"]["metric"]) == ["eff_global.wei"]
    blocks = tables["blocks_network"]
    assert list(blocks.columns[:2]) == ["network_a", "network_b"] and len(blocks) == 6
    n = len(dataset.nodes_df)
    edges = tables["edges"]
    assert list(edges.columns[:2]) == ["roi_a", "roi_b"] and len(edges) == n * (n - 1) // 2
    # the toy data has stronger within-network connectivity in the second group
    within = blocks[blocks["network_a"] == blocks["network_b"]]
    assert (within["t"] < 0).all()
    params = result.params
    assert params["groups"]["g1"] == sorted(dataset.matrices)[:3] and params["options"]["seed"] == 0
    assert params["options"]["compare"] == ["metrics", "blocks", "edges"]


def test_written_comparison_reads_back_unchanged(dataset, graph_result, tmp_path):
    from conngraph.cli.group import _read_comparison, _write_comparison

    result = compare_groups(_labels(dataset), ("A", "B"), graph_result, dataset.matrices,
                            compare=("metrics", "blocks", "edges"), n_perms=10, seed=0, verbose=False)
    _write_comparison(result, tmp_path)
    loaded = _read_comparison(tmp_path)
    assert list(loaded.tables) == list(result.tables)
    for name, table in result.tables.items():
        pd.testing.assert_frame_equal(loaded.tables[name], table, check_dtype=False)
    assert loaded.params == json.loads(json.dumps(result.params))


def test_edges_are_fisher_z_transformed_unless_asked_not_to(dataset):
    labels = _labels(dataset)
    z = compare_groups(labels, ("A", "B"), matrices=dataset.matrices, compare=["edges"], n_perms=10, seed=0,
                       verbose=False).tables["edges"]
    r = compare_groups(labels, ("A", "B"), matrices=dataset.matrices, compare=["edges"], n_perms=10, seed=0,
                       apply_fisher_z=False, verbose=False).tables["edges"]
    first = sorted(dataset.matrices)[:3]
    a, b = z.loc[0, "roi_a"], z.loc[0, "roi_b"]
    values = [dataset.matrices[s].loc[a, b] for s in first]
    assert z.loc[0, "mean_group1"] == pytest.approx(np.mean(np.arctanh(values)))
    assert r.loc[0, "mean_group1"] == pytest.approx(np.mean(values))


def test_text_covariates_become_indicator_columns_and_missing_values_drop_the_participant(dataset, graph_result):
    labels = _labels(dataset)
    ids = sorted(labels)
    cov = pd.DataFrame({"age": [20, 31, 25, 40, 22, np.nan], "site": ["x", "y", "x", "y", "y", "x"]}, index=ids)
    with pytest.warns(UserWarning, match=ids[5]):
        result = compare_groups(labels, ("A", "B"), graph_result, compare=["metrics"], covariates=cov, n_perms=10,
                                seed=0, verbose=False)
    params = result.params
    assert params["groups"]["g2"] == ids[3:5] and params["left_out"]["covariates"] == [ids[5]]
    assert params["options"]["covariates"] == ["age", "site"] and params["covariate_columns"] == ["age", "site_y"]


def test_covariates_collinear_with_the_groups_are_refused(dataset, graph_result):
    labels = _labels(dataset)
    cov = pd.DataFrame({"dx": [labels[s] for s in sorted(labels)]}, index=sorted(labels))
    with pytest.raises(ValueError, match="collinear"):
        compare_groups(labels, ("A", "B"), graph_result, compare=["metrics"], covariates=cov, n_perms=10, verbose=False)


def test_blocks_need_the_network_level(dataset):
    node_only = compute_graph_metrics(dataset.matrices, dataset.nodes_df, level="node", metrics=["strength"],
                                      graph_method="density", graph_params={"density": 0.3}, network_col="network",
                                 n_jobs=1, verbose=False)
    with pytest.raises(ValueError, match="network level"):
        compare_groups(_labels(dataset), ("A", "B"), node_only, compare=["blocks"], n_perms=10, verbose=False)


def test_participants_without_data_are_left_out_with_a_warning(dataset, graph_result):
    labels = dict(_labels(dataset), extra="B")
    with pytest.warns(UserWarning, match="extra"):
        result = compare_groups(labels, ("A", "B"), graph_result, compare=["metrics"], n_perms=10, seed=0,
                                verbose=False)
    assert result.params["left_out"]["no_data"] == ["extra"]


def test_a_drawn_seed_is_recorded(dataset, graph_result):
    result = compare_groups(_labels(dataset), ("A", "B"), graph_result, compare=["metrics"], n_perms=10, verbose=False)
    assert isinstance(result.params["options"]["seed"], int)


@pytest.mark.parametrize("correction, column", [("fdr", "p_fdr"), ("fwe", "p_fwe"), ("none", "p")])
def test_the_chosen_correction_marks_the_significant_rows(correction, column):
    values, group1 = _family()
    values["c1"] = 1.0
    table = permuted_t_test(values, group1, n_perms=50, seed=0, correction=correction, alpha=0.05)
    assert table["significant"].tolist() == (table[column] < 0.05).tolist()
    assert table.loc["c0", "significant"] and not table.loc["c1", "significant"]


def test_compare_groups_records_the_correction(dataset, graph_result):
    result = compare_groups(_labels(dataset), ("A", "B"), graph_result, compare=["metrics"], n_perms=10, seed=0,
                            verbose=False, correction="fwe", alpha=0.01)
    assert result.params["options"]["correction"] == "fwe" and result.params["options"]["alpha"] == 0.01
    assert all("significant" in t for t in result.tables.values())
    without = compare_groups(_labels(dataset), ("A", "B"), graph_result, compare=["metrics"], n_perms=10, seed=0,
                             verbose=False)
    assert not any("significant" in t for t in without.tables.values())
    with pytest.raises(ValueError, match="correction"):
        compare_groups(_labels(dataset), ("A", "B"), graph_result, compare=["metrics"], verbose=False,
                       correction="bonferroni")


def _sections(text):
    import re

    return re.findall(r'<h2 id="([^"]+)"', text)


@pytest.mark.parametrize("correction, words", [("fdr", "false discovery rate p &lt; 0.05"),
                                                ("fwe", "family-wise error-corrected p &lt; 0.05"),
                                                ("none", "uncorrected p &lt; 0.05")])
def test_comparison_report_lists_results_and_writes_the_methods(dataset, graph_result, tmp_path, correction, words):
    result = compare_groups(_labels(dataset), ("A", "B"), graph_result, dataset.matrices,
                            compare=("metrics", "blocks", "edges"), n_perms=20, seed=0, verbose=False,
                            correction=correction)
    path = tmp_path / "compare.html"
    result.save_report(path, nodes=dataset.nodes_df, network_col="network")
    text = path.read_text(encoding="utf-8")
    assert _sections(text) == ["Summary", "Metrics", "Global", "Blocks", "Edges", "Errors", "Methods", "Versions"]
    assert words in text and "Fisher z-transformed" in text and "A vs B" in text
    assert text.count('class="plotly-graph-div"') >= 3  # t heatmaps of the blocks and the edges


def test_comparison_report_notes_untested_values_and_left_out_participants(dataset, graph_result, tmp_path):
    labels = dict(_labels(dataset), extra="B")
    with pytest.warns(UserWarning):
        result = compare_groups(labels, ("A", "B"), graph_result, compare=["metrics"], n_perms=10, seed=0,
                                verbose=False, correction="fdr")
    result.params["untested"] = {"metrics_node": ["strength.abs / Def_L_00"]}
    path = tmp_path / "compare.html"
    result.save_report(path)
    text = path.read_text(encoding="utf-8")
    assert "Def_L_00" in text and "extra" in text


def test_comparison_report_needs_a_correction(dataset, graph_result, tmp_path):
    result = compare_groups(_labels(dataset), ("A", "B"), graph_result, compare=["metrics"], n_perms=10, seed=0,
                            verbose=False)
    with pytest.raises(ValueError, match="correction"):
        result.save_report(tmp_path / "compare.html")
