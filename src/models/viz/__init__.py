# path: src/models/viz/__init__.py
"""Visualization utilities for DUET model diagnostics."""

from src.models.viz.frequency_viz import (
    plot_fft_split_single,
    plot_fft_split_batch,
    plot_energy_curve,
    plot_gradient_comparison,
    plot_fusion_weights,
    plot_uncertainty_heatmap,
    compute_gradient_magnitude,
    tensor_to_numpy,
    normalize_for_display,
)

__all__ = [
    'plot_fft_split_single',
    'plot_fft_split_batch',
    'plot_energy_curve',
    'plot_gradient_comparison',
    'plot_fusion_weights',
    'plot_uncertainty_heatmap',
    'compute_gradient_magnitude',
    'tensor_to_numpy',
    'normalize_for_display',
]
