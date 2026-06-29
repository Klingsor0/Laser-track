"""
Main Streamlit Application Entry Point
Handles app configuration, session state initialization, and tab rendering
"""

import streamlit as st
from config.theme import apply_theme, THEME
from tabs import (tab1_setup, tab2_preprocessing, tab3_edge_detection,
                  tab4_analysis, tab5_validation, tab6_esp32_camera, 
                  tab7_line_tracking, tab9_batch_acquisition, tab10_angle_analysis)

# ============= PAGE CONFIGURATION =============
st.set_page_config(
    page_title="Image Processing Pipeline",
    page_icon="🔬",
    layout="wide"
)

# ============= APPLY CUSTOM THEME =============
apply_theme()

# ============= SESSION STATE INITIALIZATION =============
# Initialize all session state variables on first run
# This prevents KeyError when accessing session state in tabs

if 'original_image' not in st.session_state:
    st.session_state.original_image = None

if 'calibration' not in st.session_state:
    st.session_state.calibration = None

if 'roi_image' not in st.session_state:
    st.session_state.roi_image = None

if 'roi_mask' not in st.session_state:
    st.session_state.roi_mask = None

if 'roi_points' not in st.session_state:
    st.session_state.roi_points = None

if 'processed_image' not in st.session_state:
    st.session_state.processed_image = None

if 'edges_image' not in st.session_state:
    st.session_state.edges_image = None

if 'optimized_snake' not in st.session_state:
    st.session_state.optimized_snake = None

if 'initial_snake' not in st.session_state:
    st.session_state.initial_snake = None

if 'energy_history' not in st.session_state:
    st.session_state.energy_history = None

if 'filtered_image' not in st.session_state:
    st.session_state.filtered_image = None
 
if 'esp32_connected' not in st.session_state:
    st.session_state.esp32_connected = False
 
if 'esp32_stream' not in st.session_state:
    st.session_state.esp32_stream = None
 
if 'esp32_captured_frame' not in st.session_state:
    st.session_state.esp32_captured_frame = None

if 'line_tracking_edges' not in st.session_state:
    st.session_state.line_tracking_edges = None
 
if 'detected_lines' not in st.session_state:
    st.session_state.detected_lines = None
 
if 'line_viz_image' not in st.session_state:
    st.session_state.line_viz_image = None
 
if 'line_tracking_snake' not in st.session_state:
    st.session_state.line_tracking_snake = None
 
if 'line_tracking_energy' not in st.session_state:
    st.session_state.line_tracking_energy = None
 
if 'line_tracking_initial' not in st.session_state:
    st.session_state.line_tracking_initial = None
 

# ============= MAIN TITLE =============
st.title("🖼️ Image Processing Pipeline")

# ============= TAB CREATION =============
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab9, tab10= st.tabs([
    "1️⃣ Setup",
    "2️⃣ Preprocessing",
    "3️⃣ Edge Detection",
    "4️⃣ Analysis",
    "5️⃣ Validation",
    "📹 ESP32 Camera",
    "📏 Line Tracking",
    "Batch Acquisition",
    "Angle Analysis"
])

# ============= RENDER TABS =============
# Each tab is rendered by calling its render() function
# This keeps the main file clean and delegates logic to individual tab modules

with tab1:
    tab1_setup.render()

with tab2:
    tab2_preprocessing.render()

with tab3:
    tab3_edge_detection.render()

with tab4:
    tab4_analysis.render()

with tab5:
    tab5_validation.render()
 
with tab6:
    tab6_esp32_camera.render()

with tab7:
    tab7_line_tracking.render()
 
with tab9:
    tab9_batch_acquisition.render()

with tab10:
    tab10_angle_analysis.render()

 
