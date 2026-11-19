"""Visualization and publication figure generation suite."""

from visualization.paper_figures import (
    configure_publication_style,
    plot_fig2_dataset_distribution,
    plot_fig3_slope_chart_per_grade_recall,
    plot_fig4_waterfall_qwk_decomposition,
    plot_fig5_lodo_heatmap,
    plot_fig6_latent_severity_axis,
    plot_fig7_referable_dr_roc,
    plot_fig8_real_xai,
    save_fig_dual,
    save_figure,
)

__all__ = [
    "configure_publication_style",
    "save_fig_dual",
    "save_figure",
    "plot_fig2_dataset_distribution",
    "plot_fig3_slope_chart_per_grade_recall",
    "plot_fig4_waterfall_qwk_decomposition",
    "plot_fig5_lodo_heatmap",
    "plot_fig6_latent_severity_axis",
    "plot_fig7_referable_dr_roc",
    "plot_fig8_real_xai",
]
