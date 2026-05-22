"""
Tab 4: Analysis - Curve Fitting and Model Comparison
Performs spline fitting, finds critical points, and compares with theoretical models
"""

import streamlit as st
import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline

from utils.curve_analysis import (
    quadratic_spline_roots,
    model_branch,
    rss_model,
    min_rss,
    fit_spline_and_derivatives,
    find_critical_points
)
from utils.plotting import create_curve_analysis_plot
from utils.models import modelo_parabolico
from utils.plots import plot_save_fig_profile_PR
from config.theme import THEME


def render():
    """
    Render Tab 4 content
    Analyzes extracted curve data:
    1. Load curve coordinates
    2. Fit splines and find critical points
    3. Compare with parabolic and branch models
    4. Export scaled analysis
    """
    
    st.header("Curve Analysis & Model Fitting")
    
    # ============= LOAD CURVE DATA =============
    st.subheader("1️⃣ Load Curve Data")
    
    # Option 1: Use optimized snake from Tab 3
    if st.session_state.optimized_snake is not None:
        if st.button("📊 Use curve from Tab 3", use_container_width=True):
            df = pd.DataFrame(st.session_state.optimized_snake, columns=['x', 'y'])
            st.session_state.analysis_data = df
            st.success("✅ Loaded curve from Tab 3")
    
    # Option 2: Upload CSV file
    uploaded_csv = st.file_uploader(
        "Or upload curve coordinates CSV",
        type=['csv'],
        help="CSV file with 'x' and 'y' columns"
    )
    
    if uploaded_csv is not None:
        df = pd.read_csv(uploaded_csv)
        st.session_state.analysis_data = df
        st.success(f"✅ Loaded {len(df)} points from CSV")
    
    # ============= PERFORM ANALYSIS =============
    if 'analysis_data' in st.session_state and st.session_state.analysis_data is not None:
        df = st.session_state.analysis_data
        
        st.subheader("2️⃣ Spline Fitting & Critical Points")
        
        # Extract coordinates
        x = df['x'].values
        y = (max(df['y']) - df['y']).values  # Flip y-axis
        
        # Fit spline and derivatives
        y_spl, y_spl_1, y_spl_2 = fit_spline_and_derivatives(x, y, k=4, s=None)
        
        # Find critical points
        min_pt = y_spl_1.roots()  # Minimum (where derivative = 0)
        critic_pts = quadratic_spline_roots(y_spl_2)  # Inflection points
        
        # Display critical point info
        col_info1, col_info2 = st.columns(2)
        
        with col_info1:
            st.metric("Minimum at x =", f"{min_pt[1]:.2f} px" if len(min_pt) > 1 else "N/A")
        
        with col_info2:
            st.metric("Critical points found", len(critic_pts))
        
        # ============= MODEL FITTING =============
        st.subheader("3️⃣ Parabolic & Branch Model Fitting")
        
        # Find initial radius estimate from critical points
        R_d = abs(min_pt[1] - critic_pts[0])
        for pt in critic_pts:
            d = abs(min_pt[1] - pt)
            R_d = d if d < R_d else R_d
        
        # Create mask for parabolic region
        radius_mask = lambda z: (z >= min_pt[1] - R_d) & (z <= min_pt[1] + R_d)
        
        # Fit parabolic model to central region
        mod_parab = modelo_parabolico(x[radius_mask(x)], y[radius_mask(x)])
        
        # Define model functions for RSS minimization
        mb_1 = lambda z: mod_parab.predict(z)
        mb_2 = lambda z, r: model_branch(mod_parab._coeficientes[2], r, z - min_pt[1])
        
        # Optimize radius to minimize residual sum of squares
        with st.spinner("Optimizing radius parameter..."):
            # Declare min_pt as global for min_rss function
            # TODO: Refactor to pass as parameter
            #global min_pt as global_min_pt
            
            R_s = min_rss(mb_1, mb_2, x, y, R_d, 50.0, 200, min_pt)
        
        gamma = mod_parab._coeficientes[2]
        
        st.success(f"✅ Optimal radius: R = {R_s:.2f} px, γ = {gamma:.6f}")
        
        # ============= VISUALIZATION =============
        st.subheader("4️⃣ Visualization")
        
        # Create evaluation range
        x_range = np.linspace(min(x), max(x), 200)
        
        # Generate plot using utility function
        fig = create_curve_analysis_plot(
            x, y, x_range, y_spl, y_spl_1,
            min_pt, critic_pts, mod_parab,
            radius_mask, R_s, model_branch
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # ============= SCALED ANALYSIS =============
        if 'calibration' in st.session_state and st.session_state.calibration is not None:
            st.subheader("5️⃣ Dimensionless Analysis")
            
            # Get calibration scale
            scale = st.session_state.calibration['scale']
            unit = st.session_state.calibration['unit']
            
            st.info(f"📏 Using calibration: {scale:.6f} {unit}/pixel")
            
            # Scale coordinates to dimensionless form
            x_scaled = (x - min_pt[1]) / R_s
            y_scaled = y / (2 * gamma * R_s**2)
            
            # Theoretical model for comparison
            x_range_sc = np.linspace(-max(x_scaled), max(x_scaled), 200)
            
            def model_tot(x_arr):
                """Theoretical dimensionless profile"""
                x_arr = np.asarray(x_arr)
                y_arr = np.empty_like(x_arr, dtype=float)
                mask = np.abs(x_arr) < 1
                y_arr[mask] = (1/2) * x_arr[mask]**2
                y_arr[~mask] = 1 - (1/2) * x_arr[~mask]**-2
                return y_arr
            
            # Plot scaled data vs theory
            import plotly.graph_objects as go
            
            fig_scaled = go.Figure()
            
            # Scaled data
            fig_scaled.add_trace(go.Scatter(
                x=x_scaled, y=y_scaled,
                mode='markers',
                marker=dict(size=6, color=THEME['secondary'], opacity=0.7),
                name='Scaled data'
            ))
            
            # Theoretical curve
            fig_scaled.add_trace(go.Scatter(
                x=x_range_sc, y=model_tot(x_range_sc),
                mode='lines',
                line=dict(width=3, color=THEME['primary']),
                name='Theoretical model'
            ))
            
            # Themed layout
            fig_scaled.update_layout(
                title="Dimensionless Profile Comparison",
                xaxis_title="r / R_s",
                yaxis_title="y / (2γR²)",
                paper_bgcolor=THEME['bg'],
                plot_bgcolor=THEME['panel'],
                font=dict(color=THEME['text'])
            )
            
            st.plotly_chart(fig_scaled, use_container_width=True)
            
            # Display physical parameters
            col_param1, col_param2, col_param3 = st.columns(3)
            
            with col_param1:
                st.metric("R_s (physical)", f"{R_s * scale:.3f} {unit}")
            
            with col_param2:
                st.metric("γ (scaled)", f"{gamma / scale:.6f} {unit}⁻¹")
            
            with col_param3:
                perturbation = 4 * (gamma**2) * R_s**2
                st.metric("Perturbation factor", f"{perturbation:.4f}")
            
            # ============= EXPORT PUBLICATION FIGURE =============
            with st.sidebar:
                st.markdown("### 💾 Export Publication Figure")
                
                with st.expander("Physical Review Format", expanded=False):
                    filename = st.text_input(
                        "Filename",
                        value="curve_analysis",
                        help="Enter name without extension"
                    )
                    
                    file_format = st.radio(
                        "Format",
                        ["eps", "pdf", "tiff"],
                        horizontal=True
                    )
                    
                    dpi = st.slider("DPI", 72, 600, 300, step=50)
                    
                    if st.button("💾 Generate", use_container_width=True, type="primary"):
                        full_filename = f"{filename}.{file_format}"
                        
                        with st.spinner("Generating publication figure..."):
                            plot_save_fig_profile_PR(
                                x_scaled, y_scaled,
                                x_range_sc, model_tot,
                                full_filename, file_format
                            )
                        
                        try:
                            with open(full_filename, "rb") as file:
                                st.download_button(
                                    label=f"⬇️ Download {full_filename}",
                                    data=file,
                                    file_name=full_filename,
                                    mime=f"image/{file_format}",
                                    use_container_width=True
                                )
                            st.success("✅ Ready!")
                        except FileNotFoundError:
                            st.error(f"❌ Could not find {full_filename}")
        
        else:
            st.info("💡 Set calibration in Tab 1 to enable dimensionless analysis")
    
    else:
        st.info("👆 Load curve data to begin analysis")
