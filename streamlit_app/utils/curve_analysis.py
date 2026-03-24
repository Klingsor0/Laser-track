"""
Curve Analysis Utilities
Mathematical functions for spline fitting, critical point detection, and model comparison
"""

import numpy as np
from scipy.interpolate import UnivariateSpline


def quadratic_spline_roots(spl):
    """
    Find roots of a quadratic spline (zeros of the function)
    
    Args:
        spl: UnivariateSpline object
        
    Returns:
        numpy.ndarray: Array of root locations
        
    Note:
        Uses piecewise quadratic approximation between knots
    """
    roots = []
    knots = spl.get_knots()
    
    # For each interval between knots
    for a, b in zip(knots[:-1], knots[1:]):
        # Evaluate at three points to get quadratic coefficients
        u, v, w = spl(a), spl((a+b)/2), spl(b)
        
        # Solve quadratic: u + (w-u)*t + (u+w-2v)*t^2 = 0
        t = np.roots([u + w - 2*v, w - u, 2*v])
        
        # Keep only real roots in interval [0, 1]
        t = t[np.isreal(t) & (np.abs(t) <= 1)]
        
        # Convert from interval [0,1] to [a,b]
        roots.extend(t*(b-a)/2 + (b+a)/2)
    
    return np.array(roots)


def model_branch(a, R, x):
    """
    Branch model for curve regions outside parabolic fit
    
    Args:
        a: Parabolic coefficient (gamma)
        R: Characteristic radius
        x: x-coordinates (array-like)
        
    Returns:
        list: y-coordinates for branch model
        
    Formula:
        y = 2*a*R^2 * (1 - R^2 / (2*|x|^2))
    """
    return [2*a*R**2 * (1 - R**2 / (2*abs(xi)**2)) for xi in x]


def rss_model(function1, function2, mask, x, y):
    """
    Calculate residual sum of squares for two-part model
    
    Args:
        function1: Function for first region (callable)
        function2: Function for second region (callable)
        mask: Boolean mask function to split data
        x: Independent variable data
        y: Dependent variable data
        
    Returns:
        list: Squared residuals for both regions combined
        
    Example:
        mask = lambda z: z < threshold
        rss = rss_model(parabola, branch, mask, x_data, y_data)
    """
    # Apply functions to their respective regions
    estimated1 = function1(x[mask(x)])
    estimated2 = function2(x[~mask(x)])
    
    # Calculate squared residuals
    residues1 = [(yi[0] - yi[1])**2 for yi in zip(estimated1, y[mask(x)])]
    residues2 = [(yi[0] - yi[1])**2 for yi in zip(estimated2, y[~mask(x)])]
    
    return list(set(residues1 + residues2))


def min_rss(function1, function2, x, y, R_d, window, steps):
    """
    Find radius R that minimizes residual sum of squares
    
    Args:
        function1: Parabolic model function
        function2: Branch model function (takes x and R as args)
        x: x-coordinate data
        y: y-coordinate data
        R_d: Initial guess for radius
        window: Search range around R_d
        steps: Number of steps in search grid
        
    Returns:
        float: Optimal radius value
        
    Note:
        Searches in range [R_d - window, R_d + window]
        Uses min_pt from global scope (should be refactored)
    """
    # TODO: Pass min_pt as parameter instead of using global
    # This is a code smell that should be fixed
    global min_pt
    
    # Initialize with initial guess
    Rmin = R_d
    radius_mask_rmin = lambda z: (z >= min_pt[1] - Rmin) & (z <= min_pt[1] + Rmin)
    function2_r = lambda z: function2(z, Rmin)
    rss_min = sum(rss_model(function1, function2_r, radius_mask_rmin, x, y))
    
    # Grid search over radius values
    looking_domain = np.linspace(R_d - window, R_d + window, steps)
    for ri in looking_domain:
        radius_mask_ri = lambda z: (z >= min_pt[1] - ri) & (z <= min_pt[1] + ri)
        function2_r = lambda z: function2(z, ri)
        rss_i = sum(rss_model(function1, function2_r, radius_mask_ri, x, y))
        
        # Update if better fit found
        if rss_i < rss_min:
            rss_min = rss_i
            Rmin = ri
    
    return Rmin


def fit_spline_and_derivatives(x, y, k=4, s=None):
    """
    Fit univariate spline and compute derivatives
    
    Args:
        x: Independent variable
        y: Dependent variable
        k: Degree of spline (default: 4 for quintic)
        s: Smoothing factor (None for exact interpolation)
        
    Returns:
        tuple: (spline, first_derivative, second_derivative)
        
    Example:
        y_spl, y_spl_1, y_spl_2 = fit_spline_and_derivatives(x, y)
        critical_points = y_spl_1.roots()
    """
    y_spl = UnivariateSpline(x, y, s=s, k=k)
    y_spl_1 = y_spl.derivative(1)
    y_spl_2 = y_spl.derivative(2)
    
    return y_spl, y_spl_1, y_spl_2


def find_critical_points(y_spl_1, y_spl_2):
    """
    Find critical points (minima, maxima, inflection points)
    
    Args:
        y_spl_1: First derivative spline
        y_spl_2: Second derivative spline
        
    Returns:
        tuple: (extrema_points, inflection_points)
        
    Note:
        Extrema are where first derivative = 0
        Inflection points are where second derivative = 0
    """
    extrema = y_spl_1.roots()
    inflection = quadratic_spline_roots(y_spl_2)
    
    return extrema, inflection
