import numpy as np
import cv2


def initialize_snake(point_A, point_B, num_points=100):
    x_init = np.linspace(point_A[0], point_B[0], num_points)
    y_init = np.linspace(point_A[1], point_B[1], num_points)
    
    snake = np.array([x_init, y_init]).T  # Shape: (num_points, 2)
    return snake

def compute_external_force(image, snake):
    """
    Compute force pulling snake toward edges
    """
    # Compute edge map (gradient magnitude)
    gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
    edge_map = np.sqrt(gradient_x**2 + gradient_y**2)

    # Compute gradient of edge map (force field)
    fx = cv2.Sobel(edge_map, cv2.CV_64F, 1, 0, ksize=3)
    fy = cv2.Sobel(edge_map, cv2.CV_64F, 0, 1, ksize=3)

    # Sample force at each snake point
    forces = np.zeros_like(snake)
    for i, (x, y) in enumerate(snake):
        xi, yi = int(x), int(y)
        if 0 <= xi < image.shape[1] and 0 <= yi < image.shape[0]:
            forces[i, 0] = fx[yi, xi]
            forces[i, 1] = fy[yi, xi]

    return forces

def compute_internal_force(snake, alpha=0.01, beta=0.1):
    """
    Compute forces for smoothness and curvature

    alpha: weight for first derivative (elasticity/stretching)
    beta: weight for second derivative (rigidity/bending)
    """
    n = len(snake)
    forces = np.zeros_like(snake)

    for i in range(1, n-1):  # Skip endpoints (they're fixed)
        # First derivative (elasticity): penalize unequal spacing
        # v'(i) ≈ v[i+1] - v[i-1]
        d1 = snake[i+1] - snake[i-1]

        # Second derivative (curvature): penalize bending
        # v''(i) ≈ v[i+1] - 2*v[i] + v[i-1]
        d2 = snake[i+1] - 2*snake[i] + snake[i-1]

        # Force to minimize these
        forces[i] = alpha * d1 - beta * d2

    return forces

def compute_internal_energy(snake, alpha=0.01, beta=0.1):
    """
    Compute forces for smoothness and curvature

    alpha: weight for first derivative (elasticity/stretching)
    beta: weight for second derivative (rigidity/bending)
    """
    n = len(snake)
    forces = np.zeros_like(snake)

    for i in range(1, n-1):  # Skip endpoints (they're fixed)
        # First derivative (elasticity): penalize unequal spacing
        # v'(i) ≈ v[i+1] - v[i-1]
        elastic = (snake[i+1][0] - snake[i-1][0])**2 +  (snake[i+1][1] - snake[i-1][1])**2

        bend = (snake[i+1][0] - 2*snake[i][0] + snake[i-1][0])**2 + (snake[i+1][0] - 2*snake[i][0] + snake[i-1][0])**2 

        # Force to minimize these
        forces[i] = alpha * d1 - beta * d2

    return forces


def optimize_snake(image, point_A, point_B,
                   num_points=100,
                   num_iterations=100,
                   alpha=0.01, beta=0.1, gamma=1.0,
                   step_size=0.05):
    """
    Main optimization loop

    gamma: weight for external force
    """
    snake = initialize_snake(point_A, point_B, num_points)

    for iteration in range(num_iterations):
        # Compute forces
        f_internal = compute_internal_force(snake, alpha, beta)
        f_external = compute_external_force(image, snake)

        # Total force
        f_total = f_internal + gamma * f_external

        # Update snake positions (gradient descent)
        snake[1] += step_size * f_total

        # CRITICAL: Keep endpoints fixed
        snake[0] = point_A
        snake[-1] = point_B

        # Optional: enforce function constraint (one y per x)
        # snake = enforce_function_constraint(snake)

    return snake

def initialize_snake_from_polygon(canvas_points, num_points=100):
    """
    Create snake from user-drawn polygon points
    Interpolate to get smooth curve with num_points
    """
    # Extract x, y from canvas points
    x_poly = np.array([p[0] for p in canvas_points])
    y_poly = np.array([p[1] for p in canvas_points])
    
    # Interpolate along the polygon
    # Create parameter t from 0 to 1 along the polygon
    distances = np.cumsum(np.sqrt(np.diff(x_poly)**2 + np.diff(y_poly)**2))
    distances = np.insert(distances, 0, 0)
    t_poly = distances / distances[-1]
    
    # Interpolate to get smooth curve
    from scipy.interpolate import interp1d
    fx = interp1d(t_poly, x_poly, kind='linear')
    fy = interp1d(t_poly, y_poly, kind='linear')
    
    t_smooth = np.linspace(0, 1, num_points)
    x_smooth = fx(t_smooth)
    y_smooth = fy(t_smooth)
    
    snake = np.array([x_smooth, y_smooth]).T
    return snake

def optimize_snake_function(image, initial_snake,
                            num_iterations=100,
                            alpha=0.01, beta=0.1, gamma=1.0,
                            step_size=0.1):
    """
    Optimize snake where x coordinates are FIXED
    Only y values move
    """
    snake = initial_snake.copy()
    x_fixed = snake[:, 0].copy()  # Save x coordinates

    # Precompute edge forces
    gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    edge_map = np.sqrt(gradient_x**2 + gradient_y**2)

    # Gradient of edge map (force field in y direction)
    fy = cv2.Sobel(edge_map, cv2.CV_64F, 0, 1, ksize=3)

    n = len(snake)

    for iteration in range(num_iterations):
        y_forces = np.zeros(n)

        # Internal forces (only in y direction)
        for i in range(1, n-1):
            # Elasticity: penalize unequal y spacing
            d1_y = snake[i+1, 1] - snake[i-1, 1]

            # Rigidity: penalize curvature in y
            d2_y = snake[i+1, 1] - 2*snake[i, 1] + snake[i-1, 1]

            y_forces[i] = alpha * d1_y - beta * d2_y

        # External forces (edge attraction)
        for i in range(n):
            xi, yi = int(snake[i, 0]), int(snake[i, 1])
            if 0 <= xi < image.shape[1] and 0 <= yi < image.shape[0]:
                y_forces[i] += gamma * fy[yi, xi]

        # Update ONLY y coordinates
        snake[:, 1] += step_size * y_forces

        # Keep x fixed
        snake[:, 0] = x_fixed

        # Keep endpoints fixed (optional)
        snake[0] = initial_snake[0]
        snake[-1] = initial_snake[-1]

    return snake
