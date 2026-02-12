import numpy as np
import cv2


def compute_point_energy(snake, i, image, alpha=0.01, beta=0.1, gamma=1.0):
    """
    Compute total energy at point i
    
    Energy = alpha*elastic + beta*bending - gamma*potential
    (We want HIGH potential, so negative sign)
    """
    n = len(snake)
    
    # Skip endpoints
    if i == 0 or i == n-1:
        return 0
    
    # Elastic energy (stretching)
    elastic = ((snake[i+1][0] - snake[i-1][0])**2 + 
               (snake[i+1][1] - snake[i-1][1])**2)
    
    # Bending energy (curvature)
    # Note: You had a typo - second term should use [1] not [0]
    bend = ((snake[i+1][0] - 2*snake[i][0] + snake[i-1][0])**2 + 
            (snake[i+1][1] - 2*snake[i][1] + snake[i-1][1])**2)
    
    # External potential (edge strength)
    x, y = int(snake[i][0]), int(snake[i][1])
    if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
        # Gradient magnitude at this point
        potential = image[y, x]
    else:
        potential = 0
    
    # Total energy (negative potential because we want to maximize it)
    energy = alpha * elastic + beta * bend + gamma * potential
    
    return energy

def find_best_position(snake, i, image, window_size=5,
                       alpha=0.01, beta=0.1, gamma=1.0):
    """
    Search in a local window around snake[i] for position with lowest energy

    Returns: (best_x, best_y, energy_improvement)
    """
    current_x, current_y = snake[i]
    current_energy = compute_point_energy(snake, i, image, alpha, beta, gamma)

    best_x, best_y = current_x, current_y
    best_energy = current_energy

    # Search in window
    for dx in range(-window_size, window_size+1):
        for dy in range(-window_size, window_size+1):
            # Try new position
            new_x = current_x + dx
            new_y = current_y + dy

            # Check bounds
            if not (0 <= new_x < image.shape[1] and 0 <= new_y < image.shape[0]):
                continue

            # Temporarily move point
            old_pos = snake[i].copy()
            snake[i] = [new_x, new_y]

            # Compute energy at new position
            new_energy = compute_point_energy(snake, i, image, alpha, beta, gamma)

            # Keep if better
            if new_energy < best_energy:
                best_energy = new_energy
                best_x, best_y = new_x, new_y

            # Restore
            snake[i] = old_pos

    energy_improvement = current_energy - best_energy
    return best_x, best_y, energy_improvement

def optimize_snake_greedy(image, initial_snake,
                         num_iterations=100,
                         window_size=5,
                         alpha=0.01, beta=0.1, gamma=1.0,
                         threshold=0.01):
    """
    Greedy optimization: move each point to locally best position

    threshold: stop if total energy improvement < threshold
    """
    # Precompute edge map
    gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    edge_map = np.sqrt(gradient_x**2 + gradient_y**2)

    # Normalize edge map to [0, 1]
    edge_map = edge_map / (edge_map.max() + 1e-10)

    snake = initial_snake.copy()
    n = len(snake)

    energy_history = []

    for iteration in range(num_iterations):
        total_improvement = 0

        # Visit each point (skip endpoints if fixed)
        for i in range(1, n-1):
            best_x, best_y, improvement = find_best_position(
                snake, i, edge_map, window_size, alpha, beta, gamma
            )

            # Move to best position
            snake[i] = [best_x, best_y]
            total_improvement += improvement

        energy_history.append(total_improvement)

        # Stop if converged
        if total_improvement < threshold:
            print(f"Converged at iteration {iteration}")
            break

    return snake, energy_history

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
