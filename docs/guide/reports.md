# Reports

The reports are built from the files written to `OUTPUT`, so they always show what is on disk, and each figure and table links the files it comes from. `--reports-only` rebuilds them from those files without recomputing anything, for example after changing `--surfaces`; give the same input options as the original run, and no analysis options, which are read from the saved results.

## Figures

The reports link their static figures from the `figures/` folders, so keep those next to the reports when moving them. Brain figures are static images; `--interactive-brain` also saves each one as a rotatable 3-D page in `figures/`, linked below the image. The interactive figures and these pages load plotly from its CDN, so viewing them needs an internet connection; each has a camera button that saves it as PNG.

For publication figures at 300 dpi, use the plotting functions of the Python package (`BrainNetPlotter.plot_views`, `circos_plot`, `spring_plot`, `plot_nbs_matrices`), which save to any path at 300 dpi. The [gallery](../auto_examples/index.rst) shows them.

## The brain template

Brain figures are drawn in the fsLR 32k surfaces. `--surfaces` takes another left and right `.surf.gii`, or one skull-stripped brain volume (NIfTI or AFNI BRIK/HEAD), such as a pediatric template, whose smoothed outline is used instead. Node coordinates must be in the template's space.

## Building many reports

Participant reports are built several at a time, up to `--n-jobs`. Each one needs about 1 GB of memory, so fewer run at once when they would not fit in `--mem` (in MB; default 90% of the machine's memory). If one participant's report fails, the others are still written, the error is saved as `report-sub-<label>.err` under `OUTPUT/logs/`, and the command exits with an error.
