"""
Tab 5: Validation - Model Validation with Experimental Data
Compares theoretical predictions with experimental angle measurements
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'shared_utils'))

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from models import modelo_lineal
from plotting import create_angle_analysis_plot
from config.theme import THEME


def render():
    """
    Render Tab 5 content
    Validates theoretical model against experimental measurements:
    1. Load angle measurement data
    2. Fit linear models to different regions
    3. Compare with theoretical predictions
    4. Assess model accuracy
    """
    
    st.header("Model Validation with Experimental Data")
    
    # ============= LOAD EXPERIMENTAL DATA =============
    st.subheader("1️⃣ Load Experimental Measurements")
    
    # Option to upload validation data
    uploaded_validation = st.file_uploader(
        "Upload angle measurement CSV",
        type=['csv'],
        help="CSV with 'espejo' (mirror position) and 'angulo' (angle) columns"
    )
    
    if uploaded_validation is not None:
        angle_dat = pd.read_csv(uploaded_validation)
        st.session_state.validation_data = angle_dat
    elif 'validation_data' not in st.session_state:
        # Try to load default file
        try:
            angle_dat = pd.read_csv('pruebas/mediciones-prueba-2.csv')
            st.session_state.validation_data = angle_dat
            st.info("📊 Loaded default validation dataset")
        except:
            st.warning("⚠️ No validation data loaded. Upload a CSV file.")
            return
    else:
        angle_dat = st.session_state.validation_data
    
    # Display data preview
    with st.expander("📋 Data Preview", expanded=False):
        st.write("**Columns:**", list(angle_dat.keys()))
        st.dataframe(angle_dat, use_container_width=True)
    
    # ============= DATA ANALYSIS =============
    st.subheader("2️⃣ Piecewise Linear Fitting")
    
    # Define splitting threshold
    threshold = st.slider(
        "Region split threshold",
        min_value=float(angle_dat["espejo"].min()),
        max_value=float(angle_dat["espejo"].max()),
        value=11.2,
        step=0.1,
        help="Mirror position to split data into two regions"
    )
    
    # Split data into two regions
    region1 = angle_dat[angle_dat["espejo"] < threshold]
    region2 = angle_dat[angle_dat["espejo"] >= threshold]
    
    col_region1, col_region2 = st.columns(2)
    
    with col_region1:
        st.metric("Region 1 points", len(region1))
    
    with col_region2:
        st.metric("Region 2 points", len(region2))
    
    # Fit linear models to each region
    if len(region1) > 1 and len(region2) > 1:
        # Region 1 (below threshold)
        x_a1 = region1["espejo"].tolist()
        y_a1 = region1["angulo"].tolist()
        mod_lineal1 = modelo_lineal(x_a1, y_a1)
        
        # Region 2 (above threshold)
        x_a2 = region2["espejo"].tolist()
        y_a2 = region2["angulo"].tolist()
        mod_lineal2 = modelo_lineal(x_a2, y_a2)
        
        # Display fit parameters
        col_fit1, col_fit2 = st.columns(2)
        
        with col_fit1:
            st.metric("Region 1 slope", f"{mod_lineal1.coeficiente:.6f}")
        
        with col_fit2:
            st.metric("Region 2 slope", f"{mod_lineal2.coeficiente:.6f}")
        
        # ============= THEORETICAL MODEL COMPARISON =============
        st.subheader("3️⃣ Theoretical Model Comparison")
        
        # Define theoretical model
        def model_ang(x, Rs, gamma):
            """
            Theoretical angle model
            
            Args:
                x: Mirror position array
                Rs: Characteristic radius
                gamma: Parabolic coefficient
            
            Returns:
                Predicted angle array
            """
            x = np.asarray(x)
            y = np.empty_like(x, dtype=float)
            mask = np.abs(x) > Rs
            
            # Outer region: ~ 1/x^6
            y[mask] = (gamma**2 * Rs**8) * (5 * 3.14 / 8) / x[mask]**6
            
            # Inner region: linear
            y[~mask] = 4 * gamma * x[~mask]
            
            return y
        
        # ============= COMPARISON PLOTS =============
        st.subheader("4️⃣ Visualization")
        
        # Create comparison plots
        x_lineal = np.linspace(0, 25, 200)
        
        # Plot 1: Using fitted parameters from data
        st.write("**A) Using empirical parameters:**")
        
        fig1 = go.Figure()
        
        # Data points
        fig1.add_trace(go.Scatter(
            x=angle_dat["espejo"],
            y=angle_dat["angulo"],
            mode='markers',
            marker=dict(size=8, color=THEME['secondary'], opacity=0.7),
            name='Measured data'
        ))
        
        # Theoretical curve with empirical params
        fig1.add_trace(go.Scatter(
            x=x_lineal,
            y=model_ang(x_lineal, threshold, mod_lineal1.coeficiente / 4),
            mode='lines',
            line=dict(width=2.5, color=THEME['dark_green'], dash='dashdot'),
            name=f'Model (Rs={threshold:.1f}, γ={mod_lineal1.coeficiente/4:.4f})'
        ))
        
        # Themed layout
        fig1.update_layout(
            title="Empirical Parameter Fit",
            xaxis_title="Mirror position",
            yaxis_title="Angle",
            paper_bgcolor=THEME['bg'],
            plot_bgcolor=THEME['panel'],
            font=dict(color=THEME['text'])
        )
        
        st.plotly_chart(fig1, use_container_width=True)
        
        # Plot 2: Using parameters from Tab 4 analysis (if available)
        if 'analysis_data' in st.session_state and st.session_state.analysis_data is not None:
            st.write("**B) Using parameters from curve analysis (Tab 4):**")
            
            # This requires R_s and gamma from Tab 4
            # For now, show placeholder - should be connected to Tab 4 results
            st.info("💡 Complete curve analysis in Tab 4 to enable this comparison")
            
            # Placeholder for future implementation
            # if 'R_s' in st.session_state and 'gamma' in st.session_state:
            #     scale = st.session_state.calibration['scale']
            #     R_s_phys = st.session_state.R_s * scale
            #     gamma_phys = st.session_state.gamma / scale
            #     
            #     fig2 = create angle model plot with these params
            #     st.plotly_chart(fig2)
        
        # ============= RESIDUAL ANALYSIS =============
        with st.expander("📊 Residual Analysis", expanded=False):
            # Calculate residuals
            predictions = model_ang(
                angle_dat["espejo"].values,
                threshold,
                mod_lineal1.coeficiente / 4
            )
            
            residuals = angle_dat["angulo"].values - predictions
            
            # Residual plot
            fig_res = go.Figure()
            
            fig_res.add_trace(go.Scatter(
                x=angle_dat["espejo"],
                y=residuals,
                mode='markers',
                marker=dict(size=6, color=THEME['accent']),
                name='Residuals'
            ))
            
            # Zero line
            fig_res.add_hline(y=0, line_dash="dash", line_color=THEME['text_dim'])
            
            fig_res.update_layout(
                title="Residual Plot",
                xaxis_title="Mirror position",
                yaxis_title="Residual (measured - predicted)",
                paper_bgcolor=THEME['bg'],
                plot_bgcolor=THEME['panel'],
                font=dict(color=THEME['text'])
            )
            
            st.plotly_chart(fig_res, use_container_width=True)
            
            # Statistics
            col_stats1, col_stats2, col_stats3 = st.columns(3)
            
            with col_stats1:
                st.metric("Mean residual", f"{np.mean(residuals):.4f}")
            
            with col_stats2:
                st.metric("Std residual", f"{np.std(residuals):.4f}")
            
            with col_stats3:
                rmse = np.sqrt(np.mean(residuals**2))
                st.metric("RMSE", f"{rmse:.4f}")
        
        # ============= CONCLUSIONS =============
        st.subheader("5️⃣ Conclusions")
        
        st.write(f"""
        **Model Assessment:**
        - Region 1 slope: {mod_lineal1.coeficiente:.6f}
        - Region 2 slope: {mod_lineal2.coeficiente:.6f}
        - Split threshold: {threshold:.2f}
        """)
        
        if np.std(residuals) > 0.1:  # Arbitrary threshold
            st.warning("""
            ⚠️ **High residuals detected**
            
            The measurements show significant deviation from the theoretical model.
            This suggests:
            - Experimental uncertainties
            - Model assumptions may not fully capture the physics
            - Need for parameter refinement
            """)
        else:
            st.success("✅ Model shows good agreement with experimental data")
    
    else:
        st.error("❌ Insufficient data points in one or both regions. Adjust threshold.")
