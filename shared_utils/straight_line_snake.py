"""
Straight-Line Constrained Snake Algorithm
Optimizes snake position while maintaining straight-line constraint
Uses high rigidity parameter to force straightness
"""

import numpy as np
import cv2 as cv


def initialize_line_from_endpoints(pt1, pt2, num_points=50):
    """
    Initialize snake as straight line between two endpoints
    
    Args:
        pt1: (y, x) or (row, col) - first endpoint
        pt2: (y, x) or (row, col) - second endpoint
        num_points: Number of points along the line
        
    Returns:
        numpy.ndarray: Array of shape (num_points, 2) with (x, y) coordinates
    """
    # Create parameter t from 0 to 1
    t = np.linspace(0, 1, num_points)
    
    # Interpolate between endpoints
    # Note: converting from (row, col) to (x, y)
    x = pt1[1] + t * (pt2[1] - pt1[1])  # pt[1] is column = x
    y = pt1[0] + t * (pt2[0] - pt1[0])  # pt[0] is row = y
    
    # Return as (x, y) pairs
    return np.column_stack([x, y])


def optimize_straight_line_snake(image, initial_snake, num_iterations=100,
                                  window_size=5, alpha=0.01, beta=10.0, 
                                  gamma=0.5, threshold=0.0001):
    """
    Optimize snake with high rigidity to maintain straightness
    
    This uses the standard greedy snake algorithm but with very high beta
    to force the snake to remain straight during optimization
    
    Args:
        image: Edge-detected grayscale image
        initial_snake: Initial snake points (N, 2) array with (x, y) coords
        num_iterations: Maximum iterations
        window_size: Search window radius for each point
        alpha: Elasticity weight (continuity energy)
        beta: Rigidity weight (curvature energy) - HIGH for straight lines
        gamma: Edge attraction weight (image energy)
        threshold: Convergence threshold
        
    Returns:
        tuple: (optimized_snake, energy_history)
               optimized_snake: Final snake positions (N, 2)
               energy_history: List of energy improvements per iteration
    """
    # Import your existing snake module
    # Assuming it has a greedy optimization function
    try:
        from . import snake1 as snk
        
        # Use existing snake algorithm with high rigidity
        optimized_snake, energy_history = snk.optimize_snake_greedy(
            image, 
            initial_snake,
            num_iterations=num_iterations,
            window_size=window_size,
            alpha=alpha,
            beta=beta,  # HIGH value forces straightness
            gamma=gamma,
            threshold=threshold
        )
        
        return optimized_snake, energy_history
        
    except ImportError:
        # Fallback: simple greedy implementation if snake1 not available
        return _simple_greedy_snake(
            image, initial_snake, num_iterations,
            window_size, alpha, beta, gamma, threshold
        )


def _simple_greedy_snake(image, snake, max_iterations, window_size, 
                        alpha, beta, gamma, threshold):
    """
    Simplified greedy snake implementation (fallback)
    Moves each point to minimize local energy in search window
    
    This is a basic implementation - your snake1.py should be better
    """
    snake = snake.copy().astype(np.float64)
    n_points = len(snake)
    energy_history = []
    
    # Precompute image gradient magnitude for edge energy
    grad_x = cv.Sobel(image, cv.CV_64F, 1, 0, ksize=3)
    grad_y = cv.Sobel(image, cv.CV_64F, 0, 1, ksize=3)
    edge_map = np.sqrt(grad_x**2 + grad_y**2)
    edge_map = edge_map / (np.max(edge_map) + 1e-6)  # Normalize
    
    h, w = image.shape
    
    for iteration in range(max_iterations):
        previous_snake = snake.copy()
        total_improvement = 0
        
        # Move each point to minimize energy in local window
        for i in range(n_points):
            best_pos = snake[i].copy()
            best_energy = _compute_point_energy(
                snake, i, edge_map, alpha, beta, gamma
            )
            
            # Search in window around current position
            x_int, y_int = int(snake[i, 0]), int(snake[i, 1])
            
            for dy in range(-window_size, window_size + 1):
                for dx in range(-window_size, window_size + 1):
                    new_x = x_int + dx
                    new_y = y_int + dy
                    
                    # Check bounds
                    if new_x < 0 or new_x >= w or new_y < 0 or new_y >= h:
                        continue
                    
                    # Try this position
                    test_snake = snake.copy()
                    test_snake[i] = [new_x, new_y]
                    
                    energy = _compute_point_energy(
                        test_snake, i, edge_map, alpha, beta, gamma
                    )
                    
                    if energy < best_energy:
                        best_energy = energy
                        best_pos = [new_x, new_y]
            
            # Update to best position found
            old_energy = _compute_point_energy(snake, i, edge_map, alpha, beta, gamma)
            snake[i] = best_pos
            new_energy = _compute_point_energy(snake, i, edge_map, alpha, beta, gamma)
            total_improvement += (old_energy - new_energy)
        
        energy_history.append(total_improvement)
        
        # Check convergence
        if total_improvement < threshold:
            break
    
    return snake, energy_history


def _compute_point_energy(snake, i, edge_map, alpha, beta, gamma):
    """
    Compute energy for a single snake point
    
    Energy = alpha * E_continuity + beta * E_curvature - gamma * E_image
    """
    n = len(snake)
    
    # Continuity energy (distance to neighbors)
    if i > 0 and i < n - 1:
        # Internal point
        prev_dist = np.linalg.norm(snake[i] - snake[i-1])
        next_dist = np.linalg.norm(snake[i+1] - snake[i])
        avg_dist = (prev_dist + next_dist) / 2
        
        # Want uniform spacing
        e_continuity = (prev_dist - avg_dist)**2 + (next_dist - avg_dist)**2
    else:
        e_continuity = 0
    
    # Curvature energy (deviation from straight line)
    if i > 0 and i < n - 1:
        # Second derivative approximation
        d2 = snake[i-1] - 2*snake[i] + snake[i+1]
        e_curvature = np.dot(d2, d2)
    else:
        e_curvature = 0
    
    # Image energy (edge attraction)
    x, y = int(snake[i, 0]), int(snake[i, 1])
    h, w = edge_map.shape
    
    if 0 <= x < w and 0 <= y < h:
        e_image = edge_map[y, x]  # Higher edge values = lower energy
    else:
        e_image = 0
    
    # Total energy
    total = alpha * e_continuity + beta * e_curvature - gamma * e_image
    
    return total


def extract_line_parameters(snake_points):
    """
    Extract line parameters from optimized snake
    Uses robust line fitting to get angle and position
    
    Args:
        snake_points: Optimized snake coordinates (N, 2) with (x, y)
        
    Returns:
        dict: Line parameters including angle, slope, intercept, direction
    """
    # Use OpenCV's robust line fitting
    [vx, vy, x0, y0] = cv.fitLine(
        snake_points.astype(np.float32),
        cv.DIST_L2,  # Least squares
        0,
        0.01,
        0.01
    )
    
    # Calculate angle from direction vector
    angle_rad = np.arctan2(vy[0], vx[0])
    angle_deg = angle_rad * 180 / np.pi
    
    # Calculate slope (m) and intercept (b) for line equation y = mx + b
    if abs(vx[0]) > 1e-6:  # Not vertical
        slope = vy[0] / vx[0]
        intercept = y0[0] - slope * x0[0]
    else:  # Vertical line
        slope = np.inf
        intercept = x0[0]  # x-intercept instead
    
    return {
        'angle_deg': angle_deg,
        'angle_rad': angle_rad,
        'slope': slope,
        'intercept': intercept,
        'direction': (vx[0], vy[0]),  # Normalized direction vector
        'point_on_line': (x0[0], y0[0])  # A point the line passes through
    }


def enforce_straightness_post_optimization(snake_points):
    """
    Post-processing: project all points onto best-fit line
    Guarantees perfectly straight result
    
    Args:
        snake_points: Snake coordinates (N, 2)
        
    Returns:
        numpy.ndarray: Straightened snake points
    """
    # Fit line to points
    params = extract_line_parameters(snake_points)
    
    vx, vy = params['direction']
    x0, y0 = params['point_on_line']
    
    # Project each point onto the line
    straightened = []
    
    for point in snake_points:
        # Vector from line point to current point
        dx = point[0] - x0
        dy = point[1] - y0
        
        # Project onto line direction
        projection_length = dx * vx + dy * vy
        
        # New point on line
        new_x = x0 + projection_length * vx
        new_y = y0 + projection_length * vy
        
        straightened.append([new_x, new_y])
    
    return np.array(straightened)


def visualize_snake_result(image, initial_snake, optimized_snake, 
                           line_params=None, title="Snake Optimization"):
    """
    Create visualization of snake optimization result
    
    Args:
        image: Original image
        initial_snake: Initial snake points
        optimized_snake: Optimized snake points
        line_params: Optional line parameters from extract_line_parameters
        title: Plot title
        
    Returns:
        numpy.ndarray: RGB visualization image
    """
    # Convert to RGB if grayscale
    if len(image.shape) == 2:
        vis = cv.cvtColor(image, cv.COLOR_GRAY2RGB)
    else:
        vis = image.copy()
    
    # Draw initial snake in green
    for i in range(len(initial_snake) - 1):
        pt1 = tuple(initial_snake[i].astype(int))
        pt2 = tuple(initial_snake[i+1].astype(int))
        cv.line(vis, pt1, pt2, (0, 255, 0), 2)
    
    # Draw optimized snake in red
    for i in range(len(optimized_snake) - 1):
        pt1 = tuple(optimized_snake[i].astype(int))
        pt2 = tuple(optimized_snake[i+1].astype(int))
        cv.line(vis, pt1, pt2, (255, 0, 0), 2)
    
    # Optionally draw fitted line
    if line_params is not None:
        angle_text = f"Angle: {line_params['angle_deg']:.2f}°"
        cv.putText(vis, angle_text, (10, 30), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    return vis
