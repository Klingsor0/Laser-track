"""
Tab 2: Preprocessing - Image Filtering
Applies various filters to prepare image for edge detection
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'shared_utils'))

import streamlit as st
import cv2 as cv
import numpy as np
from PIL import Image

from image_processing import gaussian_blur
from config.theme import THEME


def render():
    """
    Render Tab 2 content
    Provides controls for:
    - Gaussian blur
    - Contrast adjustment
    - Other preprocessing filters
    """
    
    st.header("Preprocessing & Filters")
    
    # Check if ROI image exists
    if st.session_state.roi_image is not None:
        
        # Get ROI image
        roi_img = st.session_state.roi_image.copy()
        
        # Convert to grayscale for processing
        if len(roi_img.shape) == 3:
            img_gray = cv.cvtColor(roi_img, cv.COLOR_RGB2GRAY)
        else:
            img_gray = roi_img
        
        # ============= FILTER CONTROLS =============
        st.subheader("Filter Parameters")
        
        col_controls, col_preview = st.columns([1, 2])
        
        with col_controls:
            # Gaussian blur control
            apply_blur = st.checkbox("Apply Gaussian Blur", value=True)
            if apply_blur:
                kernel_size = st.slider(
                    "Kernel Size", 
                    min_value=1, 
                    max_value=15, 
                    value=5, 
                    step=2,  # Must be odd
                    help="Larger values = more blur"
                )
            
            # Contrast adjustment control
            apply_contrast = st.checkbox("Adjust Contrast", value=False)
            if apply_contrast:
                alpha = st.slider(
                    "Contrast (α)", 
                    min_value=0.5, 
                    max_value=3.0, 
                    value=1.0, 
                    step=0.1,
                    help="α > 1 increases contrast, α < 1 decreases"
                )
                beta = st.slider(
                    "Brightness (β)", 
                    min_value=-100, 
                    max_value=100, 
                    value=0,
                    help="Positive values brighten, negative values darken"
                )
            
            # Additional preprocessing option
            apply_morph = st.checkbox("Morphological Operations", value=False)
            if apply_morph:
                morph_op = st.selectbox(
                    "Operation",
                    ["Erosion", "Dilation", "Opening", "Closing"]
                )
                morph_kernel = st.slider(
                    "Kernel Size",
                    min_value=3,
                    max_value=15,
                    value=5,
                    step=2
                )
            
            st.markdown("---")
            
            # Apply filters button
            if st.button("✨ Apply Filters", type="primary", use_container_width=True):
                # Start with grayscale image
                processed = img_gray.copy()
                
                # Apply Gaussian blur
                if apply_blur:
                    processed = gaussian_blur(processed, kernel_size)
                
                # Apply contrast adjustment
                if apply_contrast:
                    processed = cv.convertScaleAbs(processed, alpha=alpha, beta=beta)
                
                # Apply morphological operations
                if apply_morph:
                    kernel = cv.getStructuringElement(
                        cv.MORPH_ELLIPSE, 
                        (morph_kernel, morph_kernel)
                    )
                    
                    if morph_op == "Erosion":
                        processed = cv.erode(processed, kernel)
                    elif morph_op == "Dilation":
                        processed = cv.dilate(processed, kernel)
                    elif morph_op == "Opening":
                        processed = cv.morphologyEx(processed, cv.MORPH_OPEN, kernel)
                    elif morph_op == "Closing":
                        processed = cv.morphologyEx(processed, cv.MORPH_CLOSE, kernel)
                
                # Save to session state
                st.session_state.processed_image = processed
                st.success("✅ Filters applied successfully!")
        
        # ============= IMAGE PREVIEW =============
        with col_preview:
            st.subheader("Preview")
            
            # Show before and after
            col_before, col_after = st.columns(2)
            
            with col_before:
                st.image(img_gray, caption="Original ROI", use_container_width=True)
            
            with col_after:
                if st.session_state.processed_image is not None:
                    st.image(
                        st.session_state.processed_image, 
                        caption="Filtered", 
                        use_container_width=True
                    )
                    st.session_state.filtered_image = st.session_state.processed_image.copy()
                else:
                    st.info("Adjust filters and click 'Apply Filters'")
        
        # ============= HISTOGRAM COMPARISON =============
        if st.session_state.processed_image is not None:
            with st.expander("📊 Histogram Comparison", expanded=False):
                import plotly.graph_objects as go
                from plotly.subplots import make_subplots
                
                # Calculate histograms
                hist_orig = cv.calcHist([img_gray], [0], None, [256], [0, 256])
                hist_proc = cv.calcHist(
                    [st.session_state.processed_image], 
                    [0], None, [256], [0, 256]
                )
                
                # Create subplot figure
                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=("Original", "Processed")
                )
                
                # Original histogram
                fig.add_trace(
                    go.Scatter(
                        y=hist_orig.flatten(), 
                        mode='lines',
                        line=dict(color=THEME['secondary']),
                        name='Original'
                    ),
                    row=1, col=1
                )
                
                # Processed histogram
                fig.add_trace(
                    go.Scatter(
                        y=hist_proc.flatten(), 
                        mode='lines',
                        line=dict(color=THEME['primary']),
                        name='Processed'
                    ),
                    row=1, col=2
                )
                
                # Update layout
                fig.update_layout(
                    height=300,
                    showlegend=False,
                    paper_bgcolor=THEME['bg'],
                    plot_bgcolor=THEME['panel'],
                    font=dict(color=THEME['text'])
                )
                
                fig.update_xaxes(title_text="Intensity", color=THEME['text'])
                fig.update_yaxes(title_text="Frequency", color=THEME['text'])
                
                st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.warning("⚠️ Please select ROI in Tab 1 (Setup) first")
