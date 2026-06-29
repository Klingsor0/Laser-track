"""
Tab 9: Batch Acquisition - Capture and Process Multiple Frames
Captures N frames from ESP32 camera at a specific distance measurement
Processes each frame to extract line angle automatically
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'shared_utils'))

import streamlit as st
import cv2 as cv
import numpy as np
import time
from datetime import datetime

from optical_experiment import CaptureSession
from angle_extraction import extract_angle_from_frame
from config.theme import THEME


def render():
    """
    Render Tab 9 content
    Main workflow:
    1. Configure session parameters (distance, frame count)
    2. Capture frames from ESP32 camera
    3. Process each frame to extract angle
    4. Display progress and results
    5. Store completed session
    """
    
    st.header("Batch Acquisition - Angle Measurements")
    
    st.markdown("""
    **Workflow:** Capture N frames at a specific distance → Extract angles automatically → Store results
    
    Perfect for collecting angle vs. distance data for optical experiments.
    """)
    
    # ============= CHECK PREREQUISITES =============
    # Need ESP32 connection
    if not st.session_state.get('esp32_connected', False):
        st.warning("⚠️ Please connect to ESP32-CAM in the 'ESP32 Camera' tab first")
        return
    
    # Initialize experiment sessions list if not exists
    if 'experiment_sessions' not in st.session_state:
        st.session_state.experiment_sessions = []
    
    # Initialize current acquisition state
    if 'acquisition_active' not in st.session_state:
        st.session_state.acquisition_active = False
    
    if 'current_session' not in st.session_state:
        st.session_state.current_session = None
    
    # ============= SECTION 1: SESSION CONFIGURATION =============
    st.subheader("1️⃣ Session Configuration")
    
    col_config1, col_config2, col_config3 = st.columns(3)
    
    with col_config1:
        distance = st.number_input(
            "Distance Measurement",
            min_value=0.0,
            value=10.0,
            step=0.1,
            format="%.2f",
            help="Physical distance for this measurement session"
        )
        
        unit = st.selectbox(
            "Unit",
            ["mm", "cm", "m", "inches"],
            index=0
        )
    
    with col_config2:
        num_frames = st.number_input(
            "Number of Frames",
            min_value=1,
            max_value=500,
            value=60,
            step=1,
            help="How many frames to capture in this session"
        )
        
        frame_interval = st.number_input(
            "Frame Interval (ms)",
            min_value=10,
            max_value=5000,
            value=100,
            step=10,
            help="Time between frame captures"
        )
    
    with col_config3:
        top_margin = st.number_input(
            "Top Margin (px)",
            min_value=1,
            max_value=100,
            value=10,
            help="Search area from top for brightest point"
        )
        
        bottom_margin = st.number_input(
            "Bottom Margin (px)",
            min_value=1,
            max_value=100,
            value=10,
            help="Search area from bottom for brightest point"
        )
    
    # ============= SECTION 2: ADVANCED PARAMETERS =============
    with st.expander("⚙️ Advanced Processing Parameters", expanded=False):
        col_adv1, col_adv2 = st.columns(2)
        
        with col_adv1:
            st.write("**Snake Parameters:**")
            snake_num_points = st.slider("Number of points", 20, 200, 50)
            snake_iterations = st.slider("Iterations", 10, 500, 100)
            snake_alpha = st.slider("Elasticity (α)", 0.0, 1.0, 0.01, 0.001)
            snake_beta = st.slider("Rigidity (β)", 0.0, 20.0, 10.0, 0.5,
                                  help="HIGH value keeps line straight")
            snake_gamma = st.slider("Edge attraction (γ)", -2.0, 2.0, 0.5, 0.1)
        
        with col_adv2:
            st.write("**Edge Detection:**")
            edge_method = st.selectbox("Method", ["canny", "sobel", "threshold"])
            
            if edge_method == "canny":
                canny_low = st.slider("Canny Low", 0, 255, 50)
                canny_high = st.slider("Canny High", 0, 255, 150)
            
            blur_kernel = st.slider("Blur Kernel", 0, 15, 3, 2)
            apply_morphology = st.checkbox("Clean edges (morphology)", value=True)
    
    # Build parameters dictionaries
    snake_params = {
        'num_points': snake_num_points,
        'num_iterations': snake_iterations,
        'window_size': 5,
        'alpha': snake_alpha,
        'beta': snake_beta,
        'gamma': snake_gamma,
        'threshold': 0.0001
    }
    
    edge_params = {
        'method': edge_method,
        'blur_kernel': blur_kernel,
        'morphology': apply_morphology
    }
    
    if edge_method == 'canny':
        edge_params['canny_low'] = canny_low
        edge_params['canny_high'] = canny_high
    
    # ============= SECTION 3: ACQUISITION CONTROL =============
    st.markdown("---")
    st.subheader("2️⃣ Capture Control")
    
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
    
    with col_btn1:
        start_clicked = st.button(
            "▶️ Start Capture",
            type="primary",
            use_container_width=True,
            disabled=st.session_state.acquisition_active
        )
    
    with col_btn2:
        stop_clicked = st.button(
            "⏹️ Stop",
            use_container_width=True,
            disabled=not st.session_state.acquisition_active
        )
    
    # Start new acquisition
    if start_clicked:
        # Create new session
        st.session_state.current_session = CaptureSession(distance, unit)
        st.session_state.current_session.roi_params = None  # Could add ROI later
        st.session_state.current_session.snake_params = snake_params
        st.session_state.current_session.extraction_params = {
            'top_margin': top_margin,
            'bottom_margin': bottom_margin,
            'edge_params': edge_params
        }
        
        st.session_state.acquisition_active = True
        st.session_state.frames_captured = 0
        st.rerun()
    
    # Stop acquisition
    if stop_clicked:
        st.session_state.acquisition_active = False
        
        # Save session if any frames were captured
        if st.session_state.current_session and st.session_state.current_session.num_frames > 0:
            st.session_state.experiment_sessions.append(st.session_state.current_session)
            st.success(f"✅ Session saved: {st.session_state.current_session.num_frames} frames captured")
        
        st.session_state.current_session = None
        st.rerun()
    
    # ============= SECTION 4: LIVE ACQUISITION =============
    if st.session_state.acquisition_active:
        st.markdown("---")
        st.subheader("3️⃣ Live Processing")
        
        session = st.session_state.current_session
        
        # Progress bar
        progress = session.num_frames / num_frames
        st.progress(progress, text=f"Progress: {session.num_frames}/{num_frames} frames")
        
        # Create placeholders for live updates
        col_preview, col_stats = st.columns([2, 1])
        
        with col_preview:
            frame_placeholder = st.empty()
        
        with col_stats:
            stats_placeholder = st.empty()
        
        # Check if we're done
        if session.num_frames >= num_frames:
            st.session_state.acquisition_active = False
            st.session_state.experiment_sessions.append(session)
            st.success(f"✅ Acquisition complete! {num_frames} frames captured")
            st.balloons()
            st.session_state.current_session = None
            st.rerun()
        
        else:
            # Capture and process one frame
            try:
                # Capture frame from ESP32
                stream = st.session_state.esp32_stream
                ret, frame = stream.read()
                
                if ret:
                    # Convert BGR to RGB
                    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                    
                    # Extract angle with diagnostics
                    result = extract_angle_from_frame(
                        frame_rgb,
                        roi_bbox=None,  # Use full frame for now
                        rotation=0.0,
                        top_margin=top_margin,
                        bottom_margin=bottom_margin,
                        snake_params=snake_params,
                        edge_params=edge_params,
                        return_diagnostics=True
                    )
                    
                    # Store frame and angle
                    session.add_frame(
                        frame_rgb,
                        result['angle_deg'],
                        result['snake_optimized']
                    )
                    
                    # Display visualization
                    frame_placeholder.image(
                        result['visualization'],
                        caption=f"Frame {session.num_frames}/{num_frames} - Angle: {result['angle_deg']:.2f}°",
                        use_container_width=True
                    )
                    
                    # Display current statistics
                    if session.num_frames > 1:
                        stats = session.get_statistics()
                        stats_placeholder.metric(
                            "Current Mean",
                            f"{stats['mean']:.2f}°",
                            delta=f"±{stats['std']:.2f}°"
                        )
                    else:
                        stats_placeholder.metric(
                            "Current Angle",
                            f"{result['angle_deg']:.2f}°"
                        )
                    
                    # Wait for interval
                    time.sleep(frame_interval / 1000.0)
                    st.rerun()
                
                else:
                    st.error("❌ Failed to capture frame from camera")
                    st.session_state.acquisition_active = False
                    st.rerun()
            
            except Exception as e:
                st.error(f"❌ Processing error: {str(e)}")
                st.session_state.acquisition_active = False
                st.rerun()
    
    # ============= SECTION 5: COMPLETED SESSIONS =============
    st.markdown("---")
    st.subheader("4️⃣ Captured Sessions")
    
    if len(st.session_state.experiment_sessions) == 0:
        st.info("No sessions captured yet. Start a capture above to begin.")
    else:
        # Display sessions table
        import pandas as pd
        
        sessions_data = []
        for i, session in enumerate(st.session_state.experiment_sessions):
            stats = session.get_statistics()
            sessions_data.append({
                'ID': i,
                'Distance': f"{session.distance} {session.unit}",
                'Frames': session.num_frames,
                'Mean Angle': f"{stats['mean']:.3f}°" if stats else "N/A",
                'Std Dev': f"{stats['std']:.3f}°" if stats else "N/A",
                'Timestamp': session.timestamp.strftime('%H:%M:%S')
            })
        
        df_sessions = pd.DataFrame(sessions_data)
        st.dataframe(df_sessions, use_container_width=True)
        
        # Session management
        col_mgmt1, col_mgmt2, col_mgmt3 = st.columns(3)
        
        with col_mgmt1:
            session_to_view = st.selectbox(
                "View session details",
                range(len(st.session_state.experiment_sessions)),
                format_func=lambda i: f"Session {i}: {st.session_state.experiment_sessions[i].distance} {st.session_state.experiment_sessions[i].unit}"
            )
        
        with col_mgmt2:
            if st.button("🗑️ Delete Selected", use_container_width=True):
                del st.session_state.experiment_sessions[session_to_view]
                st.success("Session deleted")
                st.rerun()
        
        with col_mgmt3:
            if st.button("🗑️ Clear All Sessions", use_container_width=True):
                st.session_state.experiment_sessions = []
                st.success("All sessions cleared")
                st.rerun()
        
        # Show details of selected session
        if session_to_view is not None:
            with st.expander(f"📊 Session {session_to_view} Details", expanded=False):
                session = st.session_state.experiment_sessions[session_to_view]
                stats = session.get_statistics()
                
                col_det1, col_det2, col_det3 = st.columns(3)
                
                with col_det1:
                    st.metric("Distance", f"{session.distance} {session.unit}")
                    st.metric("Frames", session.num_frames)
                
                with col_det2:
                    if stats:
                        st.metric("Mean Angle", f"{stats['mean']:.3f}°")
                        st.metric("Std Dev", f"{stats['std']:.3f}°")
                
                with col_det3:
                    if stats:
                        st.metric("SEM", f"{stats['sem']:.3f}°")
                        st.metric("95% CI", f"±{1.96*stats['sem']:.3f}°")
                
                # Quick histogram
                import plotly.graph_objects as go
                
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=session.angles,
                    nbinsx=20,
                    marker=dict(color=THEME['secondary']),
                    name='Angles'
                ))
                
                fig.update_layout(
                    title=f"Angle Distribution - Session {session_to_view}",
                    xaxis_title="Angle (degrees)",
                    yaxis_title="Frequency",
                    paper_bgcolor=THEME['bg'],
                    plot_bgcolor=THEME['panel'],
                    font=dict(color=THEME['text']),
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Export individual session
                import json
                json_data = json.dumps(session.to_dict(), indent=2)
                
                st.download_button(
                    "📥 Download Session (JSON)",
                    json_data,
                    f"session_{session_to_view}_{session.distance}{session.unit}.json",
                    "application/json",
                    use_container_width=True
                )
