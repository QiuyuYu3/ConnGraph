"""
brainnet3d — 3-D brain network visualisation built on vedo.
"""

from brainnet3d.loaders                  import load, load_group, load_gordon_atlas
from brainnet3d.core.dataset             import ConnectivityDataset
from brainnet3d.viz.plotter              import BrainNetPlotter
from brainnet3d.viz.network_graphs       import spring_plot, circos_plot, spring_plot_3d, matrix_heatmap
from brainnet3d.graph_theory.graph_utils import density_graph, threshold_graph, detect_communities
from brainnet3d.viz.atlas                import load_nifti_atlas
from brainnet3d.viz.surface              import get_fsLR_surface
from brainnet3d.viz.views                import save_three_views, save_orbit_gif, make_axis_arrows
from brainnet3d.viz.nbs_plots            import plot_nbs_matrices
from brainnet3d.graph_theory             import compute_graph_metrics, GraphMetricsResult
from brainnet3d.graph_theory.nbs         import run_nbs, NBSResult

__version__ = "0.2.0"
__all__ = [
    "load", "load_group", "load_gordon_atlas",
    "ConnectivityDataset",
    "BrainNetPlotter",
    "spring_plot", "circos_plot", "spring_plot_3d", "matrix_heatmap",
    "density_graph", "threshold_graph", "detect_communities",
    "load_nifti_atlas",
    "get_fsLR_surface",
    "save_three_views", "save_orbit_gif", "make_axis_arrows",
    "plot_nbs_matrices",
    "compute_graph_metrics", "GraphMetricsResult",
    "run_nbs", "NBSResult",
]
