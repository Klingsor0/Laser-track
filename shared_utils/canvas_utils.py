"""
Canvas Utilities
Functions for processing data from streamlit-drawable-canvas
"""

import pandas as pd
import numpy as np


def canvas_to_pts(canvas, alpha):
    """
    Extract polygon points from drawable canvas JSON data
    
    Args:
        canvas: Streamlit canvas object with JSON data
        alpha: Scaling factor used when rendering canvas
               (canvas size / original image size)
        
    Returns:
        list: List of [x, y] coordinate pairs, or None if no polygon drawn
        
    Example:
        If canvas was displayed at 50% size (alpha=0.5), this function
        scales the points back to original image coordinates
    """
    # Normalize JSON data to pandas DataFrame
    objects = pd.json_normalize(canvas.json_data["objects"])
    
    # Convert object columns to strings for evaluation
    for col in objects.select_dtypes(include=['object']).columns:
        objects[col] = objects[col].astype("str")
    
    # Check if polygon path exists
    if objects.get("path") is not None:
        # Evaluate string representation to get actual list
        polygon = eval(objects['path'][0])
        pts = []
        
        # Extract coordinate pairs from path
        for pt in polygon:
            if len(pt) > 1:
                # Scale coordinates back to original image size
                # pt format: [command, x, y, ...]
                pts.append([pt[1]/alpha, pt[2]/alpha])
        
        return pts
    else:
        return None


def validate_polygon(pts, min_points=3):
    """
    Validate that polygon has enough points
    
    Args:
        pts: List of polygon points
        min_points: Minimum required points (default: 3)
        
    Returns:
        bool: True if valid, False otherwise
    """
    if pts is None:
        return False
    return len(pts) >= min_points


def polygon_area(pts):
    """
    Calculate area of polygon using shoelace formula
    
    Args:
        pts: List of [x, y] coordinate pairs
        
    Returns:
        float: Area of polygon
    """
    if pts is None or len(pts) < 3:
        return 0
    
    pts_array = np.array(pts)
    x = pts_array[:, 0]
    y = pts_array[:, 1]
    
    # Shoelace formula
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
