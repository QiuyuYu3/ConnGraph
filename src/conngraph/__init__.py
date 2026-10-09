"""
ConnGraph: graph-theory metrics, group statistics, reports and brain figures for connectivity matrices.
"""

from conngraph.loaders                  import load, load_group, load_timeseries, load_gordon_atlas
from conngraph.connectivity             import compute_connectivity
from conngraph.derivatives              import load_xcpd, load_xcpd_flat, load_fnirs_pipe
from conngraph.core.dataset             import ConnectivityDataset
from conngraph.viz.plotter              import BrainNetPlotter
from conngraph.viz.network_graphs       import spring_plot, circos_plot, spring_plot_3d, matrix_heatmap
from conngraph.graph_theory.graph_utils import density_graph, threshold_graph, detect_communities
from conngraph.viz.atlas                import load_nifti_atlas
from conngraph.viz.surface              import get_fsLR_surface
from conngraph.viz.views                import save_three_views, save_orbit_gif, make_axis_arrows
from conngraph.viz.nbs_plots            import plot_nbs_matrices
from conngraph.graph_theory             import compute_graph_metrics, GraphMetricsResult
from conngraph.graph_theory.nbs         import run_nbs, NBSResult
from conngraph.graph_theory.compare     import compare_groups, GroupComparisonResult

__version__ = "0.2.0"
__all__ = [
    "load", "load_group", "load_timeseries", "load_gordon_atlas",
    "compute_connectivity",
    "load_xcpd", "load_xcpd_flat", "load_fnirs_pipe",
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
    "compare_groups", "GroupComparisonResult",
]
