"""
Tab 1: Setup - Upload, Calibration, and ROI Selection
Handles initial image upload, spatial calibration, and region of interest selection
"""

import streamlit as st
import numpy as np
import cv2 as cv
from PIL import Image
from streamlit_drawable_canvas import st_canvas

from utils.image_processing import load_img, crop_img
from utils.canvas_utils import canvas_to_pts
from config.theme import THEME


def render():
    """
    Render Tab 1 content
    Contains three main sections:
    1. Image upload
    2. Spatial calibration
    3. ROI selection
    """
    
    st.header("Setup: Upload, Calibration & ROI")
    
    # ============= SECTION 1: IMAGE UPLOAD =============
    st.subheader("1️⃣ Upload Image")
    uploaded = st.file_uploader("Select image to analyze", type=["jpg", "png"])
    
    if uploaded is not None:
        # Load and store image in session state
        img_rgb = load_img(uploaded)
        st.session_state.original_image = img_rgb
        img_pil = Image.fromarray(img_rgb)
        h, w = img_pil.size
        
        st.success(f"✅ Image loaded: {w}×{h} pixels")
        
        # ============= SECTION 2: SPATIAL CALIBRATION =============
        st.markdown("---")
        st.subheader("2️⃣ Spatial Calibration (Optional)")
        
        # Instructions accordion
        with st.expander("📏 How to calibrate", expanded=False):
            st.markdown("""
            **Steps:**
            1. Draw a **line** along a known distance in the image
            2. Enter the **real-world measurement** below
            3. Click **"Set Calibration"**
            
            💡 **Tip:** Choose a clear reference like a ruler, scale bar, or known object size
            """)
        
        # Layout: Canvas + Controls in columns
        col_canvas, col_controls = st.columns([2, 1])
        
        with col_canvas:
            st.write("**Draw reference line:**")
            
            # Prepare image for calibration canvas
            img_calib = img_rgb.copy()
            if len(img_calib.shape) == 2:
                img_calib = cv.cvtColor(img_calib, cv.COLOR_GRAY2RGB)
            
            img_calib_pil = Image.fromarray(img_calib)
            
            # Calibration canvas for drawing line
            canvas_calib = st_canvas(
                fill_color="rgba(0, 0, 0, 0)",
                stroke_width=3,
                stroke_color=THEME['primary'],  # Green line
                background_image=img_calib_pil,
                update_streamlit=True,
                height=img_calib_pil.height,
                width=img_calib_pil.width,
                drawing_mode="line",
                key="canvas_calibration",
            )
        
        with col_controls:
            st.write("**Parameters:**")
            
            # Reference measurement input
            reference_value = st.number_input(
                "Known distance",
                min_value=0.001,
                value=10.0,
                step=0.1,
                format="%.3f",
                help="Enter the real-world length of the line you'll draw"
            )
            
            # Unit selection
            unit = st.selectbox(
                "Unit",
                ["mm", "cm", "m", "µm", "inches"],
                index=0
            )
            
            st.markdown("---")
            
            # Process calibration line if drawn
            if canvas_calib.json_data is not None:
                objects = canvas_calib.json_data.get("objects", [])
                
                if len(objects) > 0:
                    line = objects[-1]
                    if line.get("type") == "line":
                        # Extract line endpoints
                        x1, y1 = line["x1"], line["y1"]
                        x2, y2 = line["x2"], line["y2"]
                        
                        # Calculate pixel distance
                        pixel_distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                        
                        st.metric("Line length", f"{pixel_distance:.1f} px")
                        
                        if pixel_distance > 0:
                            # Calculate scale (units per pixel)
                            scale = reference_value / pixel_distance
                            
                            st.info(f"**Scale:**\n{scale:.6f} {unit}/px")
                            
                            st.markdown("---")
                            
                            # Save calibration button
                            if st.button("🎯 Set Calibration", 
                                       type="primary", 
                                       use_container_width=True):
                                st.session_state.calibration = {
                                    'scale': scale,
                                    'unit': unit,
                                    'reference_value': reference_value,
                                    'pixel_distance': pixel_distance,
                                    'line_coords': (x1, y1, x2, y2)
                                }
                                st.success("✅ Calibration saved!")
                                st.rerun()
                            
                            # Redraw button
                            if st.button("🔄 Redraw", use_container_width=True):
                                st.rerun()
                        else:
                            st.warning("⚠️ Line too short")
                    else:
                        st.info("👆 Draw a line on the image")
                else:
                    st.info("👆 Draw a line on the image")
            else:
                st.info("👆 Draw a line on the image")
        
        # Display active calibration status
        if 'calibration' in st.session_state and st.session_state.calibration is not None:
            st.markdown("---")
            calib = st.session_state.calibration
            
            col_status, col_action = st.columns([3, 1])
            
            with col_status:
                st.success(f"""
                ✅ **Calibration Active**
                - Scale: **{calib['scale']:.6f} {calib['unit']}/pixel**
                - Reference: {calib['reference_value']:.3f} {calib['unit']} = {calib['pixel_distance']:.1f} px
                """)
            
            with col_action:
                # Remove calibration button
                if st.button("❌ Remove", use_container_width=True):
                    st.session_state.calibration = None
                    st.rerun()
        
        # ============= SECTION 3: ROI SELECTION =============
        st.markdown("---")
        st.subheader("3️⃣ Select Region of Interest (ROI)")
        st.text("Right-click to close the polygon and save it")
        
        # Scale factor for canvas display (0.5 = 50% of original size)
        alpha = 1/2
        
        # ROI selection canvas
        canvas_roi = st_canvas(
            fill_color="rgba(255, 0, 0, 0.3)",  # Red semi-transparent fill
            stroke_width=2,
            stroke_color="#ff0000",  # Red outline
            background_image=img_pil,
            update_streamlit=True,
            height=alpha * w,  # Note: w and h are swapped in PIL
            width=alpha * h,
            drawing_mode="polygon",
            display_toolbar=True,
            key="canvas_roi",
        )
        
        # Process ROI polygon if drawn
        if canvas_roi.json_data is not None:
            pts = canvas_to_pts(canvas_roi, alpha)
            
            if pts is not None:
                # Convert to integer array
                arr = [np.array(pts, 'int')]
                
                # Create binary mask from polygon
                st.session_state.roi_mask = cv.fillPoly(
                    np.zeros(img_rgb.shape, np.uint8),
                    arr,
                    [1, 1, 1]  # Fill with ones
                )
                
                # Apply mask to image and crop to bounding box
                st.session_state.roi_image = crop_img(
                    np.multiply(img_rgb, st.session_state.roi_mask),
                    arr[0]
                )
                
                # Also crop the mask itself
                st.session_state.roi_mask = crop_img(
                    st.session_state.roi_mask,
                    arr[0]
                )
                
                # Save polygon points for session export
                st.session_state.roi_points = pts
                
                if st.session_state.roi_image is not None:
                    st.success("✅ ROI mask successfully saved")
                    
                    # Optional: Display cropped ROI preview
                    with st.expander("Preview ROI", expanded=False):
                        st.image(st.session_state.roi_image, 
                               caption="Cropped ROI", 
                               use_container_width=True)
    
    else:
        st.info("👆 Upload an image to begin")
