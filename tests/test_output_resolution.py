import matplotlib.pyplot as plt
import numpy as np
import pytest
import vedo
from PIL import Image

import brainnet3d as bnv


@pytest.fixture
def graph_inputs(dataset):
    m = dataset.mean_matrix().values
    nodes = dataset.nodes_df
    return m, bnv.threshold_graph(m, threshold=0.4), nodes["label"].tolist(), nodes["network"].tolist()


@pytest.mark.parametrize("name", ["spring", "heatmap", "circos", "nbs"])
def test_2d_figures_save_at_300_dpi(graph_inputs, tmp_path, name):
    m, G, labels, nets = graph_inputs
    path = str(tmp_path / f"{name}.png")
    if name == "spring":
        bnv.spring_plot(G, labels, nets, figsize=(4, 4), save_path=path)
    elif name == "heatmap":
        bnv.matrix_heatmap(m, labels, nets, figsize=(4, 4), save_path=path)
    elif name == "circos":
        bnv.circos_plot(G, labels, nets, figsize=(4, 4), edge_style="curved", save_path=path)
    else:
        bnv.plot_nbs_matrices(m, m, np.zeros_like(m), network_labels=nets, save_path=path)
    plt.close("all")
    assert np.allclose(Image.open(path).info["dpi"], 300, atol=0.5)


def test_plot_views_default_is_300_dpi(dataset):
    fig = bnv.BrainNetPlotter(dataset, subject_id="mean").plot_views(edge_threshold=0.4)
    assert fig.dpi == pytest.approx(300)
    plt.close(fig)


def test_screenshot_file_has_twice_the_pixels(dataset, tmp_path):
    path = tmp_path / "shot.png"
    image = bnv.BrainNetPlotter(dataset, subject_id="mean").plot(edge_threshold=0.4, screenshot=str(path))
    assert np.asarray(Image.open(path)).shape[:2] == (2 * image.shape[0], 2 * image.shape[1])


def test_screenshot_file_doubles_line_widths(dataset, tmp_path, monkeypatch):
    original = vedo.Plotter.screenshot
    seen = []

    def record(self, filename="screenshot.png", scale=1, asarray=False):
        seen.append((asarray, sorted(a.properties.GetLineWidth() for a in self.objects if hasattr(a, "_endpoints"))))
        return original(self, filename, scale=scale, asarray=asarray)

    monkeypatch.setattr(vedo.Plotter, "screenshot", record)
    bnv.BrainNetPlotter(dataset, subject_id="mean").plot(edge_threshold=0.4, screenshot=str(tmp_path / "shot.png"))
    (file_widths,) = [w for asarray, w in seen if not asarray]
    (array_widths,) = [w for asarray, w in seen if asarray]
    assert array_widths and np.allclose(file_widths, 2 * np.asarray(array_widths))


def test_three_views_file_has_twice_the_pixels(tmp_path):
    path = tmp_path / "views.png"
    bnv.save_three_views([vedo.Sphere(r=10)], str(path), panel_size=(60, 50), verbose=False)
    assert Image.open(path).size == (3 * 60 * 2, 50 * 2)
