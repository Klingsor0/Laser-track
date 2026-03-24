"""
Image Processing Utilities
Contains functions for image loading, cropping, rotation, and filtering
"""

import streamlit as st
import cv2 as cv
import numpy as np
from PIL import Image


@st.cache_data
def load_img(uploaded):
    """
    Load image from Streamlit uploaded file
    
    Args:
        uploaded: Streamlit UploadedFile object
        
    Returns:
        numpy.ndarray: OpenCV image in BGR format
    """
    if uploaded is not None:
        bytes_img = uploaded.getvalue()
        img_array = np.frombuffer(bytes_img, np.uint8)
        opencv_image = cv.imdecode(img_array, cv.IMREAD_COLOR)
        return opencv_image
    return None


def crop_img(img, pts):
    """
    Crop image to bounding box of given points
    
    Args:
        img: Input image (numpy array)
        pts: List of points [(x, y), ...]
        
    Returns:
        numpy.ndarray: Cropped image
    """
    # Find bounding box corners
    x_min, y_min = pts[0]
    x_max, y_max = pts[0]
    
    # Search for min/max coordinates
    for pt in pts:
        x_min = pt[0] if pt[0] < x_min else x_min
        x_max = pt[0] if pt[0] > x_max else x_max
        y_min = pt[1] if pt[1] < y_min else y_min
        y_max = pt[1] if pt[1] > y_max else y_max
    
    # Crop image (note: numpy indexing is [y, x])
    return img[y_min:y_max, x_min:x_max]


def rotacion(img, theta=0, y=0):
    """
    Rotate image around a center point
    
    Args:
        img: Input image
        theta: Rotation angle in degrees
        y: Y-coordinate of rotation center
        
    Returns:
        numpy.ndarray: Rotated image
    """
    rows, cols, color = img.shape
    # Create rotation matrix around point ((cols-1)/2, y)
    rotate_matrix = cv.getRotationMatrix2D(((cols-1)/2.0, y), theta, 1)
    return cv.warpAffine(src=img, M=rotate_matrix, dsize=(cols, rows))


def gaussian_blur(img, k):
    """
    Apply Gaussian blur to image
    
    Args:
        img: Input image
        k: Kernel size (must be odd)
        
    Returns:
        numpy.ndarray: Blurred image
    """
    return cv.GaussianBlur(img, (k, k), 0)


def apply_pipeline(img, filters):
    """
    Apply a sequence of filter functions to an image
    
    Args:
        img: Input image
        filters: List of filter functions
        
    Returns:
        numpy.ndarray: Processed image after applying all filters
        
    Example:
        filters = [lambda x: gaussian_blur(x, 5), 
                   lambda x: cv.Canny(x, 50, 150)]
        result = apply_pipeline(img, filters)
    """
    out = img.copy()
    for f in filters:
        out = f(out)
    return out
