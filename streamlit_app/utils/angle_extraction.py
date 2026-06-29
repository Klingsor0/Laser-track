"""
Angle Extraction Pipeline
Complete end-to-end pipeline: frame → angle
Framework-agnostic for easy testing and PyQt6 migration
"""

import numpy as np
import cv2 as cv

try:
    # Try relative import (when used as module)
    from .optical_experiment import AngleExtractor
    from .straight_line_snake import (
        initialize_line_from_endpoints,
        optimize_straight_line_snake,
        extract_line_parameters,
        enforce_straightness_post_optimization
    )
except ImportError:
    # Fall back to absolute import (when run as script)
    from optical_experiment import AngleExtractor
    from straight_line_snake import (
        initialize_line_from_endpoints,
        optimize_straight_line_snake,
        extract_line_parameters,
        enforce_straightness_post_optimization
    )


def extract_angle_from_frame(frame, roi_bbox=None, rotation=0.0,
                            top_margin=10, bottom_margin=10,
                            snake_params=None, edge_params=None,
                            return_diagnostics=False):
    """
    Complete extraction pipeline: frame → angle
    
    Pipeline steps:
    1. Apply ROI (if provided)
    2. Rotate image (if needed)
    3. Convert to grayscale
    4. Detect edges (Canny)
    5. Find brightest points in top/bottom margins
    6. Initialize straight-line snake
    7. Optimize snake position
    8. Extract angle from fitted line
    
    Args:
        frame: Input image (RGB or grayscale numpy array)
        roi_bbox: ROI bounding box dict {'x': int, 'y': int, 'width': int, 'height': int}
                  If None, uses entire image
        rotation: Rotation angle in degrees (counterclockwise)
        top_margin: Number of rows from top to find brightest point
        bottom_margin: Number of rows from bottom to find brightest point
        snake_params: Dict with snake parameters:
            {
                'num_points': 50,
                'num_iterations': 100,
                'window_size': 5,
                'alpha': 0.01,
                'beta': 10.0,  # High for straight lines
                'gamma': 0.5,
                'threshold': 0.0001
            }
        edge_params: Dict with edge detection parameters:
            {
                'method': 'canny',  # or 'sobel', 'threshold'
                'canny_low': 50,
                'canny_high': 150,
                'blur_kernel': 3
            }
        return_diagnostics: If True, return detailed diagnostic information
        
    Returns:
        If return_diagnostics=False:
            angle_deg: Extracted angle in degrees
        
        If return_diagnostics=True:
            dict: {
                'angle_deg': float,
                'angle_rad': float,
                'snake_initial': ndarray,
                'snake_optimized': ndarray,
                'line_params': dict,
                'brightest_points': tuple,
                'roi_image': ndarray,
                'edge_image': ndarray,
                'visualization': ndarray (RGB image with overlays)
            }
    """
    # ============= STEP 1: APPLY ROI =============
    if roi_bbox is not None:
        x = roi_bbox['x']
        y = roi_bbox['y']
        w = roi_bbox['width']
        h = roi_bbox['height']
        roi_image = frame[y:y+h, x:x+w].copy()
    else:
        roi_image = frame.copy()
    
    # ============= STEP 2: ROTATE (if needed) =============
    if abs(rotation) > 0.01:
        roi_image = _rotate_image(roi_image, rotation)
    
    # ============= STEP 3: CONVERT TO GRAYSCALE =============
    if len(roi_image.shape) == 3:
        gray_image = cv.cvtColor(roi_image, cv.COLOR_RGB2GRAY)
    else:
        gray_image = roi_image
    
    # ============= STEP 4: EDGE DETECTION =============
    edge_image = _detect_edges(gray_image, edge_params)
    
    # ============= STEP 5: FIND BRIGHTEST POINTS =============
    top_point, bottom_point = AngleExtractor.find_brightest_points_in_margins(
        gray_image, 
        top_margin=top_margin,
        bottom_margin=bottom_margin
    )
    
    # ============= STEP 6: INITIALIZE SNAKE =============
    # Use default snake parameters if not provided
    if snake_params is None:
        snake_params = {
            'num_points': 50,
            'num_iterations': 100,
            'window_size': 5,
            'alpha': 0.01,
            'beta': 10.0,  # High rigidity for straight lines
            'gamma': 0.5,
            'threshold': 0.0001
        }
    
    initial_snake = initialize_line_from_endpoints(
        top_point, 
        bottom_point,
        num_points=snake_params['num_points']
    )
    
    # ============= STEP 7: OPTIMIZE SNAKE =============
    optimized_snake, energy_history = optimize_straight_line_snake(
        edge_image,
        initial_snake,
        num_iterations=snake_params['num_iterations'],
        window_size=snake_params['window_size'],
        alpha=snake_params['alpha'],
        beta=snake_params['beta'],
        gamma=snake_params['gamma'],
        threshold=snake_params['threshold']
    )
    
    # Optional: enforce perfect straightness
    optimized_snake = enforce_straightness_post_optimization(optimized_snake)
    
    # ============= STEP 8: EXTRACT ANGLE =============
    line_params = extract_line_parameters(optimized_snake)
    angle_deg = line_params['angle_deg']
    
    # ============= RETURN RESULTS =============
    if not return_diagnostics:
        return angle_deg
    
    else:
        # Create visualization
        vis_image = _create_visualization(
            roi_image, edge_image, 
            top_point, bottom_point,
            initial_snake, optimized_snake,
            line_params
        )
        
        return {
            'angle_deg': angle_deg,
            'angle_rad': line_params['angle_rad'],
            'snake_initial': initial_snake,
            'snake_optimized': optimized_snake,
            'line_params': line_params,
            'brightest_points': (top_point, bottom_point),
            'roi_image': roi_image,
            'edge_image': edge_image,
            'visualization': vis_image,
            'energy_history': energy_history
        }


def _rotate_image(image, angle_deg):
    """
    Rotate image by given angle (counterclockwise)
    
    Args:
        image: Input image
        angle_deg: Rotation angle in degrees
        
    Returns:
        Rotated image
    """
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    
    # Get rotation matrix
    rotation_matrix = cv.getRotationMatrix2D(center, angle_deg, 1.0)
    
    # Rotate image
    rotated = cv.warpAffine(image, rotation_matrix, (w, h))
    
    return rotated


def _detect_edges(gray_image, edge_params=None):
    """
    Detect edges in grayscale image
    
    Args:
        gray_image: Grayscale input image
        edge_params: Edge detection parameters dict
        
    Returns:
        Binary edge image
    """
    if edge_params is None:
        edge_params = {
            'method': 'canny',
            'canny_low': 50,
            'canny_high': 150,
            'blur_kernel': 3
        }
    
    # Apply Gaussian blur first to reduce noise
    blur_kernel = edge_params.get('blur_kernel', 3)
    if blur_kernel > 0:
        blurred = cv.GaussianBlur(gray_image, (blur_kernel, blur_kernel), 0)
    else:
        blurred = gray_image
    
    # Edge detection method
    method = edge_params.get('method', 'canny')
    
    if method == 'canny':
        low = edge_params.get('canny_low', 50)
        high = edge_params.get('canny_high', 150)
        edges = cv.Canny(blurred, low, high)
        
    elif method == 'sobel':
        # Sobel edge detection
        grad_x = cv.Sobel(blurred, cv.CV_64F, 1, 0, ksize=3)
        grad_y = cv.Sobel(blurred, cv.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        edges = (magnitude > edge_params.get('sobel_threshold', 50)).astype(np.uint8) * 255
        
    elif method == 'threshold':
        # Simple thresholding
        _, edges = cv.threshold(blurred, edge_params.get('threshold_value', 128), 255, cv.THRESH_BINARY)
        
    else:
        raise ValueError(f"Unknown edge detection method: {method}")
    
    # Optional: morphological operations to clean edges
    if edge_params.get('morphology', False):
        kernel = np.ones((3, 3), np.uint8)
        edges = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel)
    
    return edges


def _create_visualization(original_image, edge_image, 
                         top_point, bottom_point,
                         initial_snake, optimized_snake,
                         line_params):
    """
    Create diagnostic visualization image
    
    Args:
        original_image: Original ROI image
        edge_image: Edge-detected image
        top_point: (y, x) brightest point in top margin
        bottom_point: (y, x) brightest point in bottom margin
        initial_snake: Initial snake coordinates
        optimized_snake: Optimized snake coordinates
        line_params: Line parameters dict
        
    Returns:
        RGB visualization image
    """
    # Convert to RGB if needed
    if len(original_image.shape) == 2:
        vis = cv.cvtColor(original_image, cv.COLOR_GRAY2RGB)
    else:
        vis = original_image.copy()
    
    # Draw brightest points as circles
    cv.circle(vis, (top_point[1], top_point[0]), 5, (0, 255, 255), -1)  # Cyan
    cv.circle(vis, (bottom_point[1], bottom_point[0]), 5, (0, 255, 255), -1)
    
    # Draw initial snake in green
    for i in range(len(initial_snake) - 1):
        pt1 = tuple(initial_snake[i].astype(int))
        pt2 = tuple(initial_snake[i+1].astype(int))
        cv.line(vis, pt1, pt2, (0, 255, 0), 1)
    
    # Draw optimized snake in red (thicker)
    for i in range(len(optimized_snake) - 1):
        pt1 = tuple(optimized_snake[i].astype(int))
        pt2 = tuple(optimized_snake[i+1].astype(int))
        cv.line(vis, pt1, pt2, (255, 0, 0), 2)
    
    # Add angle text
    angle_text = f"Angle: {line_params['angle_deg']:.2f} deg"
    cv.putText(vis, angle_text, (10, 30), 
              cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    return vis


def batch_extract_angles(frames, roi_bbox=None, rotation=0.0,
                         top_margin=10, bottom_margin=10,
                         snake_params=None, edge_params=None,
                         progress_callback=None):
    """
    Extract angles from multiple frames (batch processing)
    
    Args:
        frames: List of image frames
        roi_bbox: ROI bounding box (same for all frames)
        rotation: Rotation angle (same for all frames)
        top_margin: Top margin size
        bottom_margin: Bottom margin size
        snake_params: Snake parameters dict
        edge_params: Edge detection parameters dict
        progress_callback: Optional callback function(current, total)
                          called after each frame
        
    Returns:
        dict: {
            'angles': list of extracted angles,
            'diagnostics': list of diagnostic dicts (if requested),
            'failed_frames': list of frame indices that failed
        }
    """
    angles = []
    failed_frames = []
    
    total_frames = len(frames)
    
    for i, frame in enumerate(frames):
        try:
            angle = extract_angle_from_frame(
                frame,
                roi_bbox=roi_bbox,
                rotation=rotation,
                top_margin=top_margin,
                bottom_margin=bottom_margin,
                snake_params=snake_params,
                edge_params=edge_params,
                return_diagnostics=False
            )
            angles.append(angle)
            
        except Exception as e:
            print(f"Warning: Frame {i} failed: {e}")
            angles.append(np.nan)  # Mark as failed
            failed_frames.append(i)
        
        # Call progress callback if provided
        if progress_callback is not None:
            progress_callback(i + 1, total_frames)
    
    return {
        'angles': angles,
        'failed_frames': failed_frames,
        'success_rate': (total_frames - len(failed_frames)) / total_frames
    }


# ============= STANDALONE TEST FUNCTION =============

def test_extraction_on_image(image_path, show_result=True):
    """
    Test extraction pipeline on a single image
    Useful for debugging and parameter tuning
    
    Args:
        image_path: Path to test image
        show_result: If True, display visualization
        
    Returns:
        Extracted angle and diagnostics
    """
    # Load image
    image = cv.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    
    # Run extraction with diagnostics
    result = extract_angle_from_frame(
        image,
        roi_bbox=None,  # Use entire image
        return_diagnostics=True
    )
    
    print(f"Extracted angle: {result['angle_deg']:.2f}°")
    print(f"Line parameters: {result['line_params']}")
    print(f"Brightest points: {result['brightest_points']}")
    
    if show_result:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(result['roi_image'], cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(result['edge_image'], cmap='gray')
        axes[1].set_title('Edge Detection')
        axes[1].axis('off')
        
        axes[2].imshow(result['visualization'])
        axes[2].set_title(f"Result: {result['angle_deg']:.2f}°")
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    return result


if __name__ == '__main__':
    # Example usage for testing
    import sys
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        test_extraction_on_image(image_path, show_result=True)
    else:
        print("Usage: python angle_extraction.py <image_path>")
        print("Example: python angle_extraction.py test_laser_line.png")
