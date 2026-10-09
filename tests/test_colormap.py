import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from conngraph.viz.colormap import labels_to_colors, values_to_colors
from conngraph.viz.nodes import build_nodes


def test_categorical_default_keeps_set3_up_to_twelve():
    labels = [f"net{i}" for i in range(12)]
    assert labels_to_colors(labels) == [plt.get_cmap("Set3")(i)[:3] for i in range(12)]


def test_categorical_default_gives_distinct_colours_beyond_twelve():
    labels = [f"net{i}" for i in range(13)]
    assert len(set(labels_to_colors(labels))) == 13


def test_default_beyond_twelve_takes_dark_shades_first():
    tab20 = plt.get_cmap("tab20")
    colours = labels_to_colors([f"net{i}" for i in range(13)])
    assert colours == [tab20(i)[:3] for i in (0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 1, 3, 5)]


def test_explicit_tab20_keeps_its_own_order():
    tab20 = plt.get_cmap("tab20")
    assert labels_to_colors([f"net{i}" for i in range(13)], cmap="tab20") == [tab20(i)[:3] for i in range(13)]


def test_explicit_cmap_with_too_few_colours_warns():
    with pytest.warns(UserWarning, match="13 categories"):
        labels_to_colors([f"net{i}" for i in range(13)], cmap="Set3")


def test_too_few_colours_warning_points_at_caller():
    nodes = pd.DataFrame({"label": [f"r{i}" for i in range(13)], "x": 0.0, "y": 0.0, "z": 0.0,
                          "network": [f"net{i}" for i in range(13)]})
    with pytest.warns(UserWarning, match="13 categories") as record:
        build_nodes(nodes, node_color="network", node_cmap="Set3")
    assert {os.path.normcase(w.filename) for w in record} == {os.path.normcase(__file__)}


def test_category_colours_follow_sorted_names_not_input_order():
    names = ["Visual", "net10", "Default", "net2", "Auditory"]
    colours = dict(zip(names, labels_to_colors(names)))
    assert dict(zip(names[::-1], labels_to_colors(names[::-1]))) == colours
    ordered = ["Auditory", "Default", "Visual", "net2", "net10"]
    assert [colours[n] for n in ordered] == [plt.get_cmap("Set3")(i)[:3] for i in range(5)]


def test_numeric_node_colour_defaults_to_viridis():
    nodes = pd.DataFrame({"label": list("abcd"), "x": 0.0, "y": 0.0, "z": 0.0, "value": [0.0, 1.0, 2.0, 3.0]})
    spheres = build_nodes(nodes, node_color="value")
    expected = values_to_colors(nodes["value"].to_numpy(), cmap="viridis")
    np.testing.assert_allclose([s.color() for s in spheres], expected, atol=1e-6)
