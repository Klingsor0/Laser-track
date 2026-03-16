#========== ROI===========
with tab1:
    uploaded = st.file_uploader("imagen para analizar", type=["jpg", "png"])

    st.markdown("---")
    st.subheader("Spatial Calibration")

    # Instructions
    with st.expander("📏 Calibration Instructions", expanded=False):
        st.markdown("""
        **How to calibrate:**
        1. Draw a **line** along a known distance in your upload
        2. Enter the **real-world measurement**
        3. Click **"Set Calibration"**

        💡 Choose a clear reference (ruler, known object size, scale bar)
        """)

    # Layout: Canvas + Controls
    col_canvas, col_controls = st.columns([2, 1])

    with col_canvas:
        st.write("**Draw reference line on ROI:**")

        # Prepare ROI image for canvas
        if st.session_state.original_image is not None:
            roi_for_calib = st.session_state.original_image.copy()
            if len(roi_for_calib.shape) == 2:
                roi_for_calib = cv.cvtColor(roi_for_calib, cv.COLOR_GRAY2RGB)

            img_calib_pil = Image.fromarray(roi_for_calib)

            # Calibration canvas
            canvas_calib = st_canvas(
                fill_color="rgba(0, 0, 0, 0)",
                stroke_width=3,
                stroke_color="#6EBA31",
                background_image=img_calib_pil,
                update_streamlit=True,
                height=img_calib_pil.height,
                width=img_calib_pil.width,
                drawing_mode="line",
                key="canvas_calibration",
            )

            with col_controls:
                st.write("**Parameters:**")

                # Reference measurement
                reference_value = st.number_input(
                    "Known distance",
                    min_value=0.001,
                    value=10.0,
                    step=0.1,
                    format="%.3f"
                )

                unit = st.selectbox(
                    "Unit",
                    ["mm", "cm", "m", "µm", "inches"],
                    index=0
                )

                st.markdown("---")

                # Calculate if line drawn
                if canvas_calib.json_data is not None:
                    objects = canvas_calib.json_data.get("objects", [])

                    if len(objects) > 0:
                        # Get last drawn line
                        line = objects[-1]
                        if line.get("type") == "line":
                            x1, y1 = line["x1"], line["y1"]
                            x2, y2 = line["x2"], line["y2"]

                            pixel_distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

                            st.metric("Line length", f"{pixel_distance:.1f} px")

                            if pixel_distance > 0:
                                scale = reference_value / pixel_distance

                                st.info(f"**Scale:**\n{scale:.6f} {unit}/px")

                                st.markdown("---")

                                # Buttons
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

                                if st.button("🔄 Redraw", use_container_width=True):
                                    st.rerun()
                            else:
                                st.warning("Line too short")
                        else:
                            st.info("👆 Draw a line")
                    else:
                        st.info("👆 Draw a line")
                else:
                    st.info("👆 Draw a line")

    # Display current calibration
        if 'calibration' in st.session_state and st.session_state.calibration is not None:
            st.markdown("---")
            calib = st.session_state.calibration

            col_status, col_actions = st.columns([3, 1])

            with col_status:
                st.success(f"""
                ✅ **Calibration Active**
                - Scale: **{calib['scale']:.6f} {calib['unit']}/pixel**
                - Reference: {calib['reference_value']:.3f} {calib['unit']} = {calib['pixel_distance']:.1f} px
                """)

    # SELECCIÓN DE ROI
    # https://github.com/SunOner/streamlit-drawable-canvas
    st.subheader("Select ROI and calibration")
    st.text("Right click for closing the polygon and saving it")


    alpha = 1/2 # factor de reduccion
    if uploaded is not None:

        img_rgb = load_img(uploaded)
        st.session_state.original_image = img_rgb
        img_pil = Image.fromarray(img_rgb)
        h,w = img_pil.size
        # el tamaño de la imagen importa ya que en términos del nuevo tamaño que pongamos estarán exportadas en el json las coordenadas de los vértices
        canvas = st_canvas(
            fill_color="rgba(255, 0, 0, 0.3)",
            stroke_width=2,
            stroke_color="#ff0000",
            background_image=img_pil, 
            update_streamlit=True,
            height=alpha*w ,
            width=alpha*h ,
            drawing_mode="polygon",
            display_toolbar=True,
            key="canvas",
        )

        if canvas.json_data is not None:
            pts =  canvas_to_pts(canvas, alpha)
            if pts is not None:
                arr = [np.array(pts,'int')]
                #st.write(arr)
                #st.write(img_rgb.shape)
                #st.write(img_rgb.shape)
                st.session_state.roi_mask = cv.fillPoly(np.zeros(img_rgb.shape,np.uint8),arr,[1,1,1])
                st.session_state.roi_image = crop_img(np.multiply(img_rgb,st.session_state.roi_mask), arr[0])
                st.session_state.roi_mask = crop_img(st.session_state.roi_mask, arr[0])
                if st.session_state.roi_image is not None:
                    st.write("ROI mask succesfully saved")
                #st.image(img_roi_mask)
                # tenemos que cargar la imagen en un formato para que OpenCV pueda procesarla

