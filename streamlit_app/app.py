"""
Main Streamlit Application Entry Point
Handles app configuration, session state initialization, and tab rendering
"""

import streamlit as st
from config.theme import apply_theme, THEME
from tabs import tab1_setup, tab2_preprocessing, tab3_edge_detection, tab4_analysis, tab5_validation

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


# ============= MAIN TITLE =============
st.title("🖼️ Image Processing Pipeline")

# ============= TAB CREATION =============
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1️⃣ Setup",
    "2️⃣ Preprocessing",
    "3️⃣ Edge Detection",
    "4️⃣ Analysis",
    "5️⃣ Validation"
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
