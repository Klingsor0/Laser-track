"""
Tab 10: Angle Analysis - Visualization and Statistical Analysis
Analyzes captured sessions with histograms, statistics, and comparisons
Exports data for further analysis
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.optical_experiment import ExperimentData
from config.theme import THEME


def render():
    """
    Render Tab 10 content
    Provides:
    1. Session selection and viewing
    2. Histogram visualization
    3. Statistical analysis
    4. Angle vs Distance plotting
    5. Data export
    """
    
    st.header("Angle Analysis & Visualization")
    
    # Check if we have any sessions
    if 'experiment_sessions' not in st.session_state or len(st.session_state.experiment_sessions) == 0:
        st.info("📊 No data to analyze yet. Capture sessions in the 'Batch Acquisition' tab first.")
        return
    
    sessions = st.session_state.experiment_sessions
    
    # ============= SECTION 1: SESSION SELECTION =============
    st.subheader("1️⃣ Session Selection")
    
    col_select1, col_select2 = st.columns([2, 1])
    
    with col_select1:
        # Create session selector
        session_options = [
            f"Session {i}: {s.distance} {s.unit} ({s.num_frames} frames)"
            for i, s in enumerate(sessions)
        ]
        
        selected_idx = st.selectbox(
            "Select session to analyze",
            range(len(sessions)),
            format_func=lambda i: session_options[i]
        )
    
    with col_select2:
        view_mode = st.radio(
            "View Mode",
            ["Single Session", "All Sessions"],
            horizontal=False
        )
    
    # ============= SECTION 2: SINGLE SESSION ANALYSIS =============
    if view_mode == "Single Session":
        session = sessions[selected_idx]
        stats = session.get_statistics()
        
        st.markdown("---")
        st.subheader("2️⃣ Statistical Summary")
        
        # Display statistics in metrics
        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
        
        with col_stat1:
            st.metric("Mean Angle", f"{stats['mean']:.4f}°")
            st.metric("Median", f"{stats['median']:.4f}°")
        
        with col_stat2:
            st.metric("Std Dev (σ)", f"{stats['std']:.4f}°")
            st.metric("SEM", f"{stats['sem']:.4f}°")
        
        with col_stat3:
            st.metric("95% CI", f"±{1.96*stats['sem']:.4f}°")
            st.metric("Range", f"{stats['max']-stats['min']:.4f}°")
        
        with col_stat4:
            st.metric("Min", f"{stats['min']:.4f}°")
            st.metric("Max", f"{stats['max']:.4f}°")
        
        # ============= SECTION 3: HISTOGRAM =============
        st.markdown("---")
        st.subheader("3️⃣ Angle Distribution")
        
        col_hist1, col_hist2 = st.columns([3, 1])
        
        with col_hist2:
            num_bins = st.slider("Number of bins", 10, 50, 20)
            show_gaussian = st.checkbox("Show Gaussian fit", value=True)
            show_stats_on_plot = st.checkbox("Show statistics", value=True)
        
        with col_hist1:
            # Create histogram
            fig = go.Figure()
            
            # Histogram
            fig.add_trace(go.Histogram(
                x=session.angles,
                nbinsx=num_bins,
                marker=dict(
                    color=THEME['secondary'],
                    line=dict(color=THEME['primary'], width=1)
                ),
                name='Measured Angles',
                opacity=0.7
            ))
            
            # Add Gaussian fit overlay if requested
            if show_gaussian:
                x_range = np.linspace(stats['min'], stats['max'], 200)
                gaussian = (1 / (stats['std'] * np.sqrt(2*np.pi))) * \
                          np.exp(-0.5 * ((x_range - stats['mean']) / stats['std'])**2)
                
                # Scale Gaussian to histogram
                bin_width = (stats['max'] - stats['min']) / num_bins
                gaussian_scaled = gaussian * session.num_frames * bin_width
                
                fig.add_trace(go.Scatter(
                    x=x_range,
                    y=gaussian_scaled,
                    mode='lines',
                    line=dict(color=THEME['accent'], width=3),
                    name='Gaussian Fit'
                ))
            
            # Add mean line
            fig.add_vline(
                x=stats['mean'],
                line_dash="dash",
                line_color=THEME['primary'],
                annotation_text=f"Mean: {stats['mean']:.3f}°",
                annotation_position="top"
            )
            
            # Layout
            fig.update_layout(
                title=f"Angle Distribution - {session.distance} {session.unit}",
                xaxis_title="Angle (degrees)",
                yaxis_title="Frequency",
                paper_bgcolor=THEME['bg'],
                plot_bgcolor=THEME['panel'],
                font=dict(color=THEME['text']),
                hovermode='x unified',
                height=500
            )
            
            # Add statistics annotation if requested
            if show_stats_on_plot:
                stats_text = f"μ = {stats['mean']:.3f}°<br>" + \
                           f"σ = {stats['std']:.3f}°<br>" + \
                           f"SEM = {stats['sem']:.3f}°<br>" + \
                           f"n = {stats['n_frames']}"
                
                fig.add_annotation(
                    text=stats_text,
                    xref="paper", yref="paper",
                    x=0.98, y=0.98,
                    showarrow=False,
                    bgcolor="rgba(48, 57, 74, 0.8)",
                    bordercolor=THEME['forest'],
                    borderwidth=1,
                    font=dict(color=THEME['text'], size=12),
                    align='right',
                    xanchor='right',
                    yanchor='top'
                )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # ============= SECTION 4: TIME SERIES (if applicable) =============
        st.markdown("---")
        st.subheader("4️⃣ Temporal Analysis")
        
        # Plot angles vs frame number
        fig_time = go.Figure()
        
        fig_time.add_trace(go.Scatter(
            y=session.angles,
            mode='lines+markers',
            marker=dict(size=4, color=THEME['secondary']),
            line=dict(color=THEME['primary'], width=1),
            name='Angle'
        ))
        
        # Add mean line
        fig_time.add_hline(
            y=stats['mean'],
            line_dash="dash",
            line_color=THEME['accent'],
            annotation_text=f"Mean: {stats['mean']:.3f}°"
        )
        
        # Add ±1 std bands
        fig_time.add_hrect(
            y0=stats['mean'] - stats['std'],
            y1=stats['mean'] + stats['std'],
            fillcolor=THEME['forest'],
            opacity=0.2,
            line_width=0,
            annotation_text="±1σ",
            annotation_position="right"
        )
        
        fig_time.update_layout(
            title="Angle vs Frame Number",
            xaxis_title="Frame Number",
            yaxis_title="Angle (degrees)",
            paper_bgcolor=THEME['bg'],
            plot_bgcolor=THEME['panel'],
            font=dict(color=THEME['text']),
            height=400
        )
        
        st.plotly_chart(fig_time, use_container_width=True)
        
        # ============= SECTION 5: RAW DATA TABLE =============
        with st.expander("📋 Raw Angle Data", expanded=False):
            df_angles = pd.DataFrame({
                'Frame': range(1, len(session.angles) + 1),
                'Angle (deg)': session.angles
            })
            st.dataframe(df_angles, use_container_width=True, height=300)
    
    # ============= MULTI-SESSION COMPARISON =============
    else:  # view_mode == "All Sessions"
        st.markdown("---")
        st.subheader("2️⃣ Multi-Session Comparison")
        
        # Collect data from all sessions
        experiment = ExperimentData(experiment_name="Current Experiment")
        for session in sessions:
            experiment.add_session(session)
        
        distances, mean_angles, std_errors = experiment.get_mean_angle_vs_distance()
        
        # ============= ANGLE VS DISTANCE PLOT =============
        st.subheader("3️⃣ Angle vs Distance")
        
        fig_comp = go.Figure()
        
        # Scatter with error bars
        fig_comp.add_trace(go.Scatter(
            x=distances,
            y=mean_angles,
            error_y=dict(
                type='data',
                array=std_errors,
                visible=True,
                color=THEME['secondary']
            ),
            mode='markers+lines',
            marker=dict(size=12, color=THEME['primary']),
            line=dict(color=THEME['secondary'], width=2),
            name='Mean Angle ± SEM'
        ))
        
        fig_comp.update_layout(
            title="Mean Angle vs Distance (All Sessions)",
            xaxis_title=f"Distance ({sessions[0].unit})",
            yaxis_title="Mean Angle (degrees)",
            paper_bgcolor=THEME['bg'],
            plot_bgcolor=THEME['panel'],
            font=dict(color=THEME['text']),
            hovermode='x unified',
            height=500
        )
        
        st.plotly_chart(fig_comp, use_container_width=True)
        
        # ============= COMPARISON TABLE =============
        st.subheader("4️⃣ Session Comparison Table")
        
        comparison_data = []
        for i, session in enumerate(sessions):
            stats = session.get_statistics()
            comparison_data.append({
                'Session': i,
                'Distance': f"{session.distance} {session.unit}",
                'Frames': session.num_frames,
                'Mean (°)': f"{stats['mean']:.4f}",
                'Std (°)': f"{stats['std']:.4f}",
                'SEM (°)': f"{stats['sem']:.4f}",
                '95% CI': f"±{1.96*stats['sem']:.4f}",
                'Range (°)': f"{stats['max']-stats['min']:.4f}"
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        st.dataframe(df_comparison, use_container_width=True)
        
        # ============= BOX PLOT COMPARISON =============
        st.subheader("5️⃣ Distribution Comparison")
        
        fig_box = go.Figure()
        
        for i, session in enumerate(sessions):
            fig_box.add_trace(go.Box(
                y=session.angles,
                name=f"{session.distance} {session.unit}",
                marker=dict(color=THEME['secondary']),
                boxmean='sd'  # Show mean and std dev
            ))
        
        fig_box.update_layout(
            title="Angle Distributions by Distance",
            xaxis_title="Distance",
            yaxis_title="Angle (degrees)",
            paper_bgcolor=THEME['bg'],
            plot_bgcolor=THEME['panel'],
            font=dict(color=THEME['text']),
            height=500,
            showlegend=False
        )
        
        st.plotly_chart(fig_box, use_container_width=True)
    
    # ============= SECTION 6: DATA EXPORT =============
    st.markdown("---")
    st.subheader("6️⃣ Export Data")
    
    col_export1, col_export2, col_export3 = st.columns(3)
    
    with col_export1:
        # Export as CSV (all sessions, all angles)
        csv_data = []
        for i, session in enumerate(sessions):
            for frame_num, angle in enumerate(session.angles, start=1):
                csv_data.append({
                    'session_id': i,
                    'distance': session.distance,
                    'unit': session.unit,
                    'frame': frame_num,
                    'angle_deg': angle
                })
        
        df_export = pd.DataFrame(csv_data)
        csv_str = df_export.to_csv(index=False)
        
        st.download_button(
            "📥 Download All Data (CSV)",
            csv_str,
            "experiment_data.csv",
            "text/csv",
            use_container_width=True
        )
    
    with col_export2:
        # Export statistics summary
        stats_data = []
        for i, session in enumerate(sessions):
            stats = session.get_statistics()
            stats_data.append({
                'session': i,
                'distance': session.distance,
                'unit': session.unit,
                'n_frames': session.num_frames,
                'mean': stats['mean'],
                'std': stats['std'],
                'sem': stats['sem'],
                'ci_95_lower': stats['ci_95_lower'],
                'ci_95_upper': stats['ci_95_upper'],
                'min': stats['min'],
                'max': stats['max'],
                'median': stats['median']
            })
        
        df_stats = pd.DataFrame(stats_data)
        stats_csv = df_stats.to_csv(index=False)
        
        st.download_button(
            "📥 Download Statistics (CSV)",
            stats_csv,
            "experiment_statistics.csv",
            "text/csv",
            use_container_width=True
        )
    
    with col_export3:
        # Export complete experiment as JSON
        experiment = ExperimentData(experiment_name="Optical Experiment")
        for session in sessions:
            experiment.add_session(session)
        
        import json
        experiment_json = json.dumps(experiment.to_dict(), indent=2)
        
        st.download_button(
            "📥 Download Experiment (JSON)",
            experiment_json,
            "experiment_complete.json",
            "application/json",
            use_container_width=True
        )
    
    # ============= SECTION 7: LOAD PREVIOUS DATA =============
    st.markdown("---")
    st.subheader("7️⃣ Load Previous Experiment")
    
    uploaded_json = st.file_uploader(
        "Upload experiment JSON file",
        type=['json'],
        help="Load previously saved experiment data"
    )
    
    if uploaded_json is not None:
        import json
        
        try:
            data = json.load(uploaded_json)
            
            # Reconstruct experiment
            from utils.optical_experiment import CaptureSession
            
            loaded_sessions = []
            for session_data in data['sessions']:
                session = CaptureSession(
                    distance=session_data['distance'],
                    unit=session_data['unit'],
                    description=session_data.get('description', '')
                )
                session.angles = session_data['angles']
                session.num_frames = session_data['num_frames']
                session.session_id = session_data['session_id']
                from datetime import datetime
                session.timestamp = datetime.fromisoformat(session_data['timestamp'])
                
                loaded_sessions.append(session)
            
            if st.button("📂 Load These Sessions", type="primary"):
                st.session_state.experiment_sessions.extend(loaded_sessions)
                st.success(f"✅ Loaded {len(loaded_sessions)} sessions!")
                st.rerun()
        
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
