"""
Image Processing Utilities
Contains functions for image loading, cropping, rotation, and filtering
"""

import cv2 as cv
import numpy as np
from PIL import Image

try:
    import streamlit as st
    _cache = st.cache_data
except ImportError:
    _cache = lambda f: f


@_cache
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


def detect_edges_canny(img, low_threshold=50, high_threshold=150,
                       blur_kernel=3, apply_morphology=False):
    """
    Canny edge detection with optional preprocessing and morphological cleanup.

    Args:
        img:              Grayscale or BGR image (color is converted automatically)
        low_threshold:    Canny lower hysteresis threshold
        high_threshold:   Canny upper hysteresis threshold
        blur_kernel:      Gaussian blur kernel size before Canny (0 = skip blur; must be odd if > 0)
        apply_morphology: If True, apply morphological closing after Canny to
                          connect broken edge segments

    Returns:
        numpy.ndarray: Binary edge image (uint8, 0 or 255), same H×W as input
    """
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) if img.ndim == 3 else img

    if blur_kernel > 0:
        k = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
        gray = cv.GaussianBlur(gray, (k, k), 0)

    edges = cv.Canny(gray, low_threshold, high_threshold)

    if apply_morphology:
        kernel = np.ones((3, 3), np.uint8)
        edges = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel)

    return edges


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Test image_processing utilities')
    parser.add_argument('image', help='Input image path')
    parser.add_argument('--blur', type=int, default=5, help='Blur kernel size (odd)')
    parser.add_argument('--rotate', type=float, default=0.0, help='Rotation angle in degrees')
    parser.add_argument('--output', default='output.png', help='Output image path')
    args = parser.parse_args()

    img = cv.imread(args.image)
    if img is None:
        print(f'ERROR: could not load {args.image}')
        raise SystemExit(1)

    h, w = img.shape[:2]
    print(f'Loaded: {args.image}  shape={img.shape}')

    # crop_img: use a centred box covering the middle third
    x0, y0 = w // 3, h // 3
    x1, y1 = 2 * w // 3, 2 * h // 3
    roi = crop_img(img, [[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    print(f'crop_img:      {img.shape} -> {roi.shape}')

    # rotacion
    rotated = rotacion(img, theta=args.rotate, y=h // 2)
    print(f'rotacion({args.rotate}deg): shape={rotated.shape}')

    # gaussian_blur (kernel must be odd and >= 1)
    k = args.blur if args.blur % 2 == 1 else args.blur + 1
    blurred = gaussian_blur(roi, k)
    print(f'gaussian_blur(k={k}): shape={blurred.shape}')

    # apply_pipeline: blur then Canny on the ROI
    pipeline = [
        lambda x: gaussian_blur(x, k),
        lambda x: cv.Canny(cv.cvtColor(x, cv.COLOR_BGR2GRAY) if x.ndim == 3 else x, 50, 150),
    ]
    edges = apply_pipeline(roi, pipeline)
    print(f'apply_pipeline (blur+Canny): shape={edges.shape}  nonzero={np.count_nonzero(edges)} px')

    cv.imwrite(args.output, blurred)
    print(f'Saved blurred ROI -> {args.output}')

    # detect_edges_canny — four combinations
    for blur_k, morph in [(0, False), (3, False), (3, True), (5, True)]:
        e = detect_edges_canny(img, low_threshold=30, high_threshold=100,
                               blur_kernel=blur_k, apply_morphology=morph)
        assert e.ndim == 2, 'Expected 2-D output'
        assert e.dtype == np.uint8, 'Expected uint8'
        print(f'detect_edges_canny(blur={blur_k}, morph={morph}): '
              f'shape={e.shape}  nonzero={np.count_nonzero(e)} px')

    # detect_edges_canny accepts color input directly
    e_color = detect_edges_canny(img, blur_kernel=3)
    e_gray  = detect_edges_canny(cv.cvtColor(img, cv.COLOR_BGR2GRAY), blur_kernel=3)
    assert np.array_equal(e_color, e_gray), 'Color and gray paths must produce identical output'
    print('detect_edges_canny color==gray path: ✓')

    print('All tests passed.')
