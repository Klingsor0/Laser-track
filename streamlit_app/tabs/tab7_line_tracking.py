"""
Tab 7: Line Tracking - Automated Line Detection and Refinement
Automatically detects straight lines using Hough Transform, then refines with snake algorithm
Designed for tracking rotating lines with fixed pivot points (e.g., pendulums, pointers)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'shared_utils'))

import streamlit as st
import cv2 as cv
import numpy as np
import pandas as pd
from PIL import Image
import plotly.graph_objects as go

from image_processing import gaussian_blur
import snake1 as snk  # Your snake module
from config.theme import THEME


def detect_lines_hough(image, edge_image, min_line_length=50, max_line_gap=10, 
                       threshold=50, rho=1, theta=np.pi/180):
    """
    Detect straight lines using Probabilistic Hough Transform
    
    Args:
        image: Original image (for visualization)
        edge_image: Edge-detected binary image
        min_line_length: Minimum line length in pixels
        max_line_gap: Maximum gap between line segments to treat as single line
        threshold: Accumulator threshold (higher = fewer, stronger lines)
        rho: Distance resolution in pixels
        theta: Angle resolution in radians
        
    Returns:
        lines: Array of detected lines [[x1, y1, x2, y2], ...]
        viz_image: Visualization with lines drawn
    """
    # Apply Probabilistic Hough Transform
    lines = cv.HoughLinesP(
        edge_image,
        rho=rho,
        theta=theta,
        threshold=threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap
    )
    
    # Create visualization
    viz_image = cv.cvtColor(image.copy(), cv.COLOR_GRAY2RGB) if len(image.shape) == 2 else image.copy()
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv.line(viz_image, (x1, y1), (x2, y2), (110, 186, 49), 2)  # Green in RGB
    
    return lines, viz_image


def merge_similar_lines(lines, angle_threshold=5, distance_threshold=20):
    """
    Merge lines that are similar in angle and position
    Useful when Hough detects multiple segments of the same physical line
    
    Args:
        lines: Array of lines [[x1, y1, x2, y2], ...]
        angle_threshold: Maximum angle difference in degrees to merge
        distance_threshold: Maximum distance between line centers to merge
        
    Returns:
        merged_lines: Array of merged lines
    """
    if lines is None or len(lines) == 0:
        return None
    
    # Calculate angle and center for each line
    line_data = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        line_data.append({
            'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
            'angle': angle, 'cx': center_x, 'cy': center_y, 'length': length
        })
    
    # Sort by length (longest first)
    line_data.sort(key=lambda x: x['length'], reverse=True)
    
    merged = []
    used = [False] * len(line_data)
    
    for i, line1 in enumerate(line_data):
        if used[i]:
            continue
        
        # Start a group with this line
        group = [line1]
        used[i] = True
        
        # Find similar lines to merge
        for j, line2 in enumerate(line_data):
            if used[j]:
                continue
            
            # Check angle similarity
            angle_diff = abs(line1['angle'] - line2['angle'])
            if angle_diff > 180:
                angle_diff = 360 - angle_diff
            
            # Check distance between centers
            dist = np.sqrt((line1['cx'] - line2['cx'])**2 + (line1['cy'] - line2['cy'])**2)
            
            if angle_diff < angle_threshold and dist < distance_threshold:
                group.append(line2)
                used[j] = True
        
        # Merge group into single line (weighted by length)
        if len(group) == 1:
            merged.append([group[0]['x1'], group[0]['y1'], group[0]['x2'], group[0]['y2']])
        else:
            # Average endpoints weighted by length
            total_length = sum(l['length'] for l in group)
            x1 = sum(l['x1'] * l['length'] for l in group) / total_length
            y1 = sum(l['y1'] * l['length'] for l in group) / total_length
            x2 = sum(l['x2'] * l['length'] for l in group) / total_length
            y2 = sum(l['y2'] * l['length'] for l in group) / total_length
            merged.append([int(x1), int(y1), int(x2), int(y2)])
    
    return np.array(merged).reshape(-1, 1, 4)


def extend_line_to_pivot(x1, y1, x2, y2, pivot_x, pivot_y, extension_factor=1.5):
    """
    Extend a line segment to pass through a pivot point
    
    Args:
        x1, y1, x2, y2: Line endpoints
        pivot_x, pivot_y: Pivot point coordinates
        extension_factor: How much to extend beyond detected endpoints
        
    Returns:
        (new_x1, new_y1, new_x2, new_y2): Extended line endpoints
    """
    # Calculate line direction vector
    dx = x2 - x1
    dy = y2 - y1
    length = np.sqrt(dx**2 + dy**2)
    
    if length == 0:
        return x1, y1, x2, y2
    
    # Normalize direction
    dx_norm = dx / length
    dy_norm = dy / length
    
    # Project pivot onto line to find closest point
    # Vector from line start to pivot
    px = pivot_x - x1
    py = pivot_y - y1
    
    # Dot product gives projection distance
    projection = px * dx_norm + py * dy_norm
    
    # Extend line in both directions from pivot
    extended_length = length * extension_factor
    new_x1 = pivot_x - dx_norm * extended_length
    new_y1 = pivot_y - dy_norm * extended_length
    new_x2 = pivot_x + dx_norm * extended_length
    new_y2 = pivot_y + dy_norm * extended_length
    
    return int(new_x1), int(new_y1), int(new_x2), int(new_y2)


def line_to_snake_points(x1, y1, x2, y2, num_points=50):
    """
    Convert line segment to array of points for snake initialization
    
    Args:
        x1, y1, x2, y2: Line endpoints
        num_points: Number of points to generate along line
        
    Returns:
        points: Array of [x, y] coordinates along the line
    """
    t = np.linspace(0, 1, num_points)
    x = x1 + t * (x2 - x1)
    y = y1 + t * (y2 - y1)
    
    return np.column_stack([x, y])


def render():
    """
    Render Tab 7 content
    Automated line detection and tracking workflow
    """
    
    st.header("Automated Line Tracking")
    
    st.markdown("""
    **Workflow:** Edge Detection → Hough Line Detection → Line Selection → Snake Refinement
    
    Perfect for tracking rotating pointers, pendulums, or any straight-line features.
    """)
    
    # Check if we have a preprocessed image
    if st.session_state.filtered_image is None:
        st.warning("⚠️ Please apply filters in Tab 2 (Preprocessing) first")
        return
    
    # ============= SECTION 1: EDGE DETECTION =============
    st.subheader("1️⃣ Edge Detection")
    
    img_filtered = st.session_state.filtered_image.copy()
    
    # Convert to grayscale if needed
    if len(img_filtered.shape) == 3:
        img_gray = cv.cvtColor(img_filtered, cv.COLOR_RGB2GRAY)
    else:
        img_gray = img_filtered
    
    col_edge1, col_edge2 = st.columns(2)
    
    with col_edge1:
        st.write("**Canny Edge Detection Parameters:**")
        
        threshold1 = st.slider(
            "Lower Threshold",
            min_value=0,
            max_value=255,
            value=50,
            help="Lower threshold for edge detection"
        )
        
        threshold2 = st.slider(
            "Upper Threshold",
            min_value=0,
            max_value=255,
            value=150,
            help="Upper threshold for edge detection"
        )
        
        # Apply Canny edge detection
        edges = cv.Canny(img_gray, threshold1, threshold2)
        
        # Optional: morphological operations to clean edges
        apply_morph = st.checkbox("Clean edges (morphology)", value=True)
        if apply_morph:
            kernel = np.ones((3, 3), np.uint8)
            edges = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel)
            edges = cv.dilate(edges, kernel, iterations=1)
    
    with col_edge2:
        st.image(edges, caption="Edge Detection Result", use_container_width=True)
    
    # Store edges in session state
    st.session_state.line_tracking_edges = edges
    
    # ============= SECTION 2: HOUGH LINE DETECTION =============
    st.markdown("---")
    st.subheader("2️⃣ Hough Line Detection")
    
    col_hough1, col_hough2 = st.columns([1, 2])
    
    with col_hough1:
        st.write("**Hough Transform Parameters:**")
        
        threshold = st.slider(
            "Accumulator Threshold",
            min_value=10,
            max_value=200,
            value=50,
            help="Higher = fewer, stronger lines detected"
        )
        
        min_line_length = st.slider(
            "Min Line Length (px)",
            min_value=10,
            max_value=500,
            value=100,
            help="Minimum length for a line to be detected"
        )
        
        max_line_gap = st.slider(
            "Max Line Gap (px)",
            min_value=1,
            max_value=50,
            value=10,
            help="Maximum gap between segments to treat as one line"
        )
        
        merge_lines = st.checkbox(
            "Merge similar lines",
            value=True,
            help="Combine line segments that are part of the same physical line"
        )
        
        if merge_lines:
            angle_threshold = st.slider("Angle threshold (°)", 1, 30, 5)
            distance_threshold = st.slider("Distance threshold (px)", 5, 100, 20)
        
        if st.button("🔍 Detect Lines", type="primary", use_container_width=True):
            # Detect lines
            detected_lines, viz_image = detect_lines_hough(
                img_gray, edges,
                min_line_length=min_line_length,
                max_line_gap=max_line_gap,
                threshold=threshold
            )
            
            # Merge if requested
            if merge_lines and detected_lines is not None:
                detected_lines = merge_similar_lines(
                    detected_lines,
                    angle_threshold=angle_threshold,
                    distance_threshold=distance_threshold
                )
            
            # Store results
            st.session_state.detected_lines = detected_lines
            st.session_state.line_viz_image = viz_image
            
            if detected_lines is not None:
                st.success(f"✅ Detected {len(detected_lines)} line(s)")
            else:
                st.warning("⚠️ No lines detected. Try adjusting parameters.")
    
    with col_hough2:
        if 'line_viz_image' in st.session_state and st.session_state.line_viz_image is not None:
            st.image(
                st.session_state.line_viz_image,
                caption=f"Detected Lines ({len(st.session_state.detected_lines) if st.session_state.detected_lines is not None else 0})",
                use_container_width=True
            )
        else:
            st.info("👈 Click 'Detect Lines' to run Hough Transform")
    
    # ============= SECTION 3: LINE SELECTION =============
    if 'detected_lines' in st.session_state and st.session_state.detected_lines is not None:
        st.markdown("---")
        st.subheader("3️⃣ Line Selection & Refinement")
        
        lines = st.session_state.detected_lines
        
        # Create line selection table
        line_data = []
        for i, line in enumerate(lines):
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            line_data.append({
                'ID': i,
                'Length (px)': f"{length:.1f}",
                'Angle (°)': f"{angle:.1f}",
                'Center': f"({center_x:.0f}, {center_y:.0f})",
                'Start': f"({x1}, {y1})",
                'End': f"({x2}, {y2})"
            })
        
        df_lines = pd.DataFrame(line_data)
        st.dataframe(df_lines, use_container_width=True)
        
        col_select, col_pivot = st.columns(2)
        
        with col_select:
            selected_line_id = st.selectbox(
                "Select line to track",
                options=range(len(lines)),
                format_func=lambda x: f"Line {x} ({line_data[x]['Length (px)']} px, {line_data[x]['Angle (°)']}°)"
            )
        
        with col_pivot:
            use_pivot = st.checkbox(
                "Use pivot point",
                value=False,
                help="If tracking a rotating line with fixed pivot (e.g., pendulum)"
            )
            
            if use_pivot:
                col_px, col_py = st.columns(2)
                with col_px:
                    pivot_x = st.number_input("Pivot X", value=int(img_gray.shape[1]/2))
                with col_py:
                    pivot_y = st.number_input("Pivot Y", value=int(img_gray.shape[0]/2))
        
        # Display selected line
        if selected_line_id is not None:
            x1, y1, x2, y2 = lines[selected_line_id][0]
            
            # Extend line to pivot if requested
            if use_pivot:
                x1, y1, x2, y2 = extend_line_to_pivot(
                    x1, y1, x2, y2,
                    pivot_x, pivot_y,
                    extension_factor=1.5
                )
            
            # Visualize selected line
            img_selected = cv.cvtColor(img_gray.copy(), cv.COLOR_GRAY2RGB)
            cv.line(img_selected, (x1, y1), (x2, y2), (255, 0, 0), 3)  # Red for selected
            
            if use_pivot:
                cv.circle(img_selected, (pivot_x, pivot_y), 8, (0, 255, 255), -1)  # Cyan pivot
            
            st.image(img_selected, caption="Selected Line for Tracking", use_container_width=True)
            
            # ============= SECTION 4: SNAKE REFINEMENT =============
            st.markdown("---")
            st.subheader("4️⃣ Snake Algorithm Refinement")
            
            col_snake1, col_snake2 = st.columns(2)
            
            with col_snake1:
                st.write("**Snake Parameters:**")
                
                num_points = st.slider(
                    "Number of points",
                    min_value=20,
                    max_value=200,
                    value=50,
                    help="Points along the line for snake algorithm"
                )
                
                alpha_energy = st.slider(
                    "Elasticity (α)",
                    0.0, 1.0, 0.01, 0.001,
                    help="Controls stretching"
                )
                
                beta_energy = st.slider(
                    "Rigidity (β)",
                    -0.2, 0.2, 0.0, 0.001,
                    help="Controls bending (keep low for straight lines)"
                )
                
                gamma_energy = st.slider(
                    "Edge attraction (γ)",
                    -2.0, 1.0, 0.5, 0.1,
                    help="Pulls snake toward edges"
                )
            
            with col_snake2:
                st.write("**Optimization Settings:**")
                
                num_iterations = st.slider("Iterations", 10, 500, 100)
                window_size = st.slider("Window size", 1, 50, 5, 1)
                threshold_conv = 0.0001 * st.slider("Convergence threshold", 0.01, 1.0, 0.1, 0.01)
            
            if st.button("🎯 Refine Line with Snake", type="primary", use_container_width=True):
                with st.spinner("Optimizing line position..."):
                    # Initialize snake from detected line
                    initial_points = line_to_snake_points(x1, y1, x2, y2, num_points)
                    
                    # Run snake optimization
                    optimized_snake, energy_history = snk.optimize_snake_greedy(
                        edges, initial_points,
                        num_iterations=num_iterations,
                        window_size=int(window_size),
                        alpha=alpha_energy,
                        beta=beta_energy,
                        gamma=gamma_energy,
                        threshold=threshold_conv
                    )
                    
                    # Store results
                    st.session_state.line_tracking_snake = optimized_snake
                    st.session_state.line_tracking_energy = energy_history
                    st.session_state.line_tracking_initial = initial_points
                
                st.success(f"✅ Refinement complete! Converged in {len(energy_history)} iterations")
            
            # ============= SECTION 5: RESULTS =============
            if 'line_tracking_snake' in st.session_state and st.session_state.line_tracking_snake is not None:
                st.markdown("---")
                st.subheader("5️⃣ Tracking Results")
                
                col_res1, col_res2 = st.columns(2)
                
                with col_res1:
                    # Show initial
                    img_init = cv.cvtColor(edges.copy(), cv.COLOR_GRAY2RGB)
                    initial_pts = st.session_state.line_tracking_initial
                    for i in range(len(initial_pts)-1):
                        pt1 = tuple(initial_pts[i].astype(int))
                        pt2 = tuple(initial_pts[i+1].astype(int))
                        cv.line(img_init, pt1, pt2, (0, 255, 0), 2)
                    st.image(img_init, caption="Initial (Hough Detection)", use_container_width=True)
                
                with col_res2:
                    # Show refined
                    img_refined = cv.cvtColor(edges.copy(), cv.COLOR_GRAY2RGB)
                    refined_pts = st.session_state.line_tracking_snake
                    for i in range(len(refined_pts)-1):
                        pt1 = tuple(refined_pts[i].astype(int))
                        pt2 = tuple(refined_pts[i+1].astype(int))
                        cv.line(img_refined, pt1, pt2, (255, 0, 0), 2)
                    st.image(img_refined, caption="Refined (Snake Algorithm)", use_container_width=True)
                
                # Calculate line parameters from refined result
                # Fit line to refined points
                [vx, vy, x0, y0] = cv.fitLine(
                    refined_pts.astype(np.float32),
                    cv.DIST_L2, 0, 0.01, 0.01
                )
                
                # Calculate angle
                angle_rad = np.arctan2(vy[0], vx[0])
                angle_deg = angle_rad * 180 / np.pi
                
                # Display metrics
                col_metric1, col_metric2, col_metric3 = st.columns(3)
                
                with col_metric1:
                    st.metric("Line Angle", f"{angle_deg:.2f}°")
                
                with col_metric2:
                    st.metric("Center Point", f"({int(x0[0])}, {int(y0[0])})")
                
                with col_metric3:
                    line_length = np.linalg.norm(refined_pts[-1] - refined_pts[0])
                    st.metric("Tracked Length", f"{line_length:.1f} px")
                
                # Energy convergence plot
                with st.expander("📊 Optimization Convergence", expanded=False):
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        y=st.session_state.line_tracking_energy,
                        mode='lines',
                        line=dict(color=THEME['primary'], width=2),
                        name='Energy improvement'
                    ))
                    
                    fig.update_layout(
                        title="Snake Optimization Convergence",
                        xaxis_title="Iteration",
                        yaxis_title="Energy Improvement",
                        paper_bgcolor=THEME['bg'],
                        plot_bgcolor=THEME['panel'],
                        font=dict(color=THEME['text'])
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Export options
                st.markdown("---")
                st.subheader("6️⃣ Export Results")
                
                col_exp1, col_exp2, col_exp3 = st.columns(3)
                
                with col_exp1:
                    # Save coordinates
                    df_coords = pd.DataFrame(refined_pts, columns=['x', 'y'])
                    csv = df_coords.to_csv(index=False)
                    
                    st.download_button(
                        "📥 Download Coordinates (CSV)",
                        csv,
                        "line_tracking_coords.csv",
                        "text/csv",
                        use_container_width=True
                    )
                
                with col_exp2:
                    # Save line parameters
                    params = {
                        'angle_deg': angle_deg,
                        'angle_rad': angle_rad,
                        'center_x': float(x0[0]),
                        'center_y': float(y0[0]),
                        'direction_x': float(vx[0]),
                        'direction_y': float(vy[0]),
                        'length': float(line_length)
                    }
                    
                    import json
                    json_str = json.dumps(params, indent=2)
                    
                    st.download_button(
                        "📥 Download Parameters (JSON)",
                        json_str,
                        "line_tracking_params.json",
                        "application/json",
                        use_container_width=True
                    )
                
                with col_exp3:
                    # Send to analysis tab
                    if st.button("📤 Send to Tab 4 (Analysis)", use_container_width=True):
                        # Convert to DataFrame for analysis
                        df = pd.DataFrame(refined_pts, columns=['x', 'y'])
                        st.session_state.analysis_data = df
                        st.success("✅ Line data sent to Analysis tab")
