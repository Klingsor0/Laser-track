"""
Utilities Package
Contains helper functions for image processing, curve analysis, and visualization
"""

from .image_processing import load_img, crop_img, rotacion, gaussian_blur, apply_pipeline
from .canvas_utils import canvas_to_pts, validate_polygon, polygon_area
from .curve_analysis import (
    quadratic_spline_roots, 
    model_branch, 
    rss_model, 
    min_rss,
    fit_spline_and_derivatives,
    find_critical_points
)
from .plotting import create_curve_analysis_plot, create_angle_analysis_plot

__all__ = [
    # Image processing
    'load_img',
    'crop_img',
    'rotacion',
    'gaussian_blur',
    'apply_pipeline',
    
    # Canvas utilities
    'canvas_to_pts',
    'validate_polygon',
    'polygon_area',
    
    # Curve analysis
    'quadratic_spline_roots',
    'model_branch',
    'rss_model',
    'min_rss',
    'fit_spline_and_derivatives',
    'find_critical_points',
    
    # Plotting
    'create_curve_analysis_plot',
    'create_angle_analysis_plot',
]
