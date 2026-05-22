"""
Tab 3: Edge Detection - Active Contour Curve Extraction
Uses greedy energy minimization to extract curves from edge-detected images
"""

import streamlit as st
import cv2 as cv
import numpy as np
import pandas as pd
from PIL import Image
from streamlit_drawable_canvas import st_canvas

from utils.canvas_utils import canvas_to_pts
from utils import snake1 as snk  # Your snake module
from config.theme import THEME


def render():
    """
    Render Tab 3 content
    Allows user to:
    1. Draw initial curve approximation
    2. Optimize curve using active contour
    3. Export extracted curve coordinates
    """
    
    st.header("Edge Detection & Curve Extraction")
    
    # Check if preprocessed image exists
    if st.session_state.filtered_image is not None:
        
        # Get filtered image and extract green channel (or convert to grayscale)
        img_filtered = st.session_state.filtered_image.copy()
        
        # Extract edge image (green channel if RGB, or grayscale)
        if len(img_filtered.shape) == 3:
            img_edges = img_filtered[:, :, 1]  # Green channel
        else:
            img_edges = img_filtered
        
        # Display edge image for reference
        with st.expander("📷 Preview Edge Image", expanded=False):
            st.image(img_edges, caption="Edge Image (Green Channel)", use_container_width=True)
        
        # ============= INITIAL CURVE DRAWING =============
        st.subheader("1️⃣ Draw Initial Curve Approximation")
        st.text("Draw a polygon along the curve you want to extract")
        
        edge_image_pil = Image.fromarray(img_edges)
        h2, w2 = edge_image_pil.size
        
        # Canvas scale factor
        alpha = 1/2
        
        st.write(h2, w2)
        # Canvas for drawing initial curve
        canvas_result = st_canvas(
            fill_color="rgba(0, 255, 0, 0.3)",  # Green semi-transparent
            stroke_width=2,
            stroke_color="#00FF00",  # Green outline
            background_image=edge_image_pil,
            update_streamlit=True,
            #height=int(alpha * w2),
            #width=int(alpha * h2),
            drawing_mode="polygon",
            display_toolbar=True,
            key="canvas_curve_init_2",
        )
        
        # ============= CURVE OPTIMIZATION =============
        if canvas_result.json_data is not None:
            pts = canvas_to_pts(canvas_result, alpha)
            
            if pts is not None and len(pts) >= 3:
                st.success(f"✅ {len(pts)} points captured")
                
                st.subheader("2️⃣ Optimization Parameters")
                
                # Parameter controls in two columns
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Energy Weights:**")
                    alpha_energy = st.slider(
                        "Elasticity (α)", 
                        0.0, 1.0, 0.01, 0.001,
                        help="Controls stretching - higher values prefer uniform spacing"
                    )
                    beta_energy = st.slider(
                        "Rigidity (β)", 
                        -0.2, 0.2, 0.1, 0.001,
                        help="Controls bending - higher values prefer smooth curves"
                    )
                    gamma_energy = st.slider(
                        "Edge attraction (γ)", 
                        -2.0, 1.0, 0.1, 0.1,
                        help="Controls edge strength - positive attracts to edges"
                    )
                
                with col2:
                    st.write("**Algorithm Settings:**")
                    num_iterations = st.slider(
                        "Iterations", 
                        10, 500, 100,
                        help="Maximum optimization iterations"
                    )
                    num_points = st.slider(
                        "Interpolated points", 
                        50, 300, 100,
                        help="Number of points in final curve"
                    )
                    window_size = st.slider(
                        "Window size", 
                        1, 50, 5, 1,
                        help="Search radius for each point"
                    )
                    threshold = 0.0001 * st.slider(
                        "Convergence threshold", 
                        0.01, 1.0, 0.1, 0.01,
                        help="Stop when energy change is below this value"
                    )
                
                # Optimize button
                if st.button("🎯 Optimize Curve", type="primary", use_container_width=True):
                    with st.spinner("Optimizing curve..."):
                        # Initialize snake from canvas points
                        snake = snk.initialize_snake_from_polygon(pts, num_points=num_points)
                        st.session_state.initial_snake = snake.copy()
                        
                        # Run greedy optimization
                        optimized_snake, energy_history = snk.optimize_snake_greedy(
                            img_edges, snake,
                            num_iterations=num_iterations,
                            window_size=int(window_size),
                            alpha=alpha_energy, 
                            beta=beta_energy, 
                            gamma=gamma_energy,
                            threshold=threshold
                        )
                        
                        # Store results in session state
                        st.session_state.optimized_snake = optimized_snake
                        st.session_state.energy_history = energy_history
                    
                    st.success(f"✅ Optimization complete! Converged in {len(energy_history)} iterations")
                
                # ============= RESULTS VISUALIZATION =============
                if st.session_state.optimized_snake is not None:
                    st.subheader("3️⃣ Results")
                    
                    # Show before/after comparison
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        # Initial approximation
                        img_init = cv.cvtColor(img_edges.copy(), cv.COLOR_GRAY2RGB)
                        for i in range(len(pts)-1):
                            pt1 = tuple(int(x) for x in pts[i])
                            pt2 = tuple(int(x) for x in pts[i+1])
                            cv.line(img_init, pt1, pt2, (0, 255, 0), 2)
                        st.image(img_init, caption="Initial approximation", use_container_width=True)
                    
                    with col_b:
                        # Optimized curve
                        img_opt = cv.cvtColor(img_edges.copy(), cv.COLOR_GRAY2RGB)
                        optimized_snake = st.session_state.optimized_snake
                        for i in range(len(optimized_snake)-1):
                            pt1 = tuple(optimized_snake[i].astype(int))
                            pt2 = tuple(optimized_snake[i+1].astype(int))
                            cv.line(img_opt, pt1, pt2, (255, 0, 0), 2)
                        st.image(img_opt, caption="Optimized curve", use_container_width=True)
                    
                    # Energy convergence plot
                    if st.session_state.energy_history is not None:
                        with st.expander("📊 Energy Convergence", expanded=False):
                            import plotly.graph_objects as go
                            
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                y=st.session_state.energy_history,
                                mode='lines',
                                line=dict(color=THEME['primary'], width=2),
                                name='Energy improvement'
                            ))
                            
                            fig.update_layout(
                                title="Optimization Convergence",
                                xaxis_title="Iteration",
                                yaxis_title="Energy Improvement",
                                paper_bgcolor=THEME['bg'],
                                plot_bgcolor=THEME['panel'],
                                font=dict(color=THEME['text'])
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                    
                    # ============= DATA EXPORT =============
                    st.subheader("4️⃣ Export Curve Coordinates")
                    
                    # Convert to DataFrame
                    df = pd.DataFrame(optimized_snake, columns=['x', 'y'])
                    
                    # Preview data
                    st.dataframe(df, height=200, use_container_width=True)
                    
                    # Export options
                    col_export1, col_export2 = st.columns(2)
                    
                    with col_export1:
                        # CSV download
                        csv = df.to_csv(index=False)
                        st.download_button(
                            "📥 Download CSV",
                            csv,
                            "curve_coordinates.csv",
                            "text/csv",
                            use_container_width=True
                        )
                    
                    with col_export2:
                        # NumPy array download
                        import io
                        buffer = io.BytesIO()
                        np.save(buffer, optimized_snake)
                        buffer.seek(0)
                        
                        st.download_button(
                            "📥 Download NPY",
                            buffer,
                            "curve_coordinates.npy",
                            "application/octet-stream",
                            use_container_width=True
                        )
            else:
                st.info("👆 Draw a polygon with at least 3 points on the image above")
    
    else:
        st.warning("⚠️ Please apply filters in Tab 2 (Preprocessing) first")
