import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from brainnet3d.viz.colormap import labels_to_colors, values_to_colors
from brainnet3d.viz.nodes import build_nodes


def test_categorical_default_keeps_set3_up_to_twelve():
    labels = [f"net{i}" for i in range(12)]
    assert labels_to_colors(labels) == [plt.get_cmap("Set3")(i)[:3] for i in range(12)]


def test_categorical_default_gives_distinct_colours_beyond_twelve():
    labels = [f"net{i}" for i in range(13)]
    assert len(set(labels_to_colors(labels))) == 13


def test_explicit_cmap_with_too_few_colours_warns():
    with pytest.warns(UserWarning, match="13 categories"):
        labels_to_colors([f"net{i}" for i in range(13)], cmap="Set3")


def test_numeric_node_colour_defaults_to_viridis():
    nodes = pd.DataFrame({"label": list("abcd"), "x": 0.0, "y": 0.0, "z": 0.0, "value": [0.0, 1.0, 2.0, 3.0]})
    spheres = build_nodes(nodes, node_color="value")
    expected = values_to_colors(nodes["value"].to_numpy(), cmap="viridis")
    np.testing.assert_allclose([s.color() for s in spheres], expected, atol=1e-6)
