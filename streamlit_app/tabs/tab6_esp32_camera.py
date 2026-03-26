"""
Tab 6: ESP32 Camera - Live Camera Stream
Streams video from ESP32-CAM over network using MJPEG
"""

import streamlit as st
import cv2 as cv
import numpy as np
import time
from PIL import Image

from config.theme import THEME


def render():
    """
    Render Tab 6 content
    Provides live streaming from ESP32-CAM with controls for:
    - Connection management
    - Frame capture
    - Stream settings
    - Image export to other tabs
    """
    
    st.header("ESP32-CAM Live Stream")
    
    # ============= SECTION 1: CONNECTION SETUP =============
    st.subheader("1️⃣ Camera Connection")
    
    col_connection, col_status = st.columns([2, 1])
    
    with col_connection:
        # ESP32 IP address input
        esp32_ip = st.text_input(
            "ESP32-CAM IP Address",
            value="192.168.1.100",
            help="Enter the IP address of your ESP32-CAM on the network"
        )
        
        # Port selection (standard ESP32-CAM uses port 81 for stream)
        esp32_port = st.number_input(
            "Port",
            min_value=1,
            max_value=65535,
            value=81,
            help="Port number for MJPEG stream (default: 81)"
        )
        
        # Stream path
        stream_path = st.text_input(
            "Stream Path",
            value="/stream",
            help="URL path for the stream endpoint"
        )
        
        # Construct full URL
        stream_url = f"http://{esp32_ip}:{esp32_port}{stream_path}"
        st.code(stream_url, language="text")
    
    with col_status:
        st.write("**Connection Status:**")
        
        # Initialize connection state
        if 'esp32_connected' not in st.session_state:
            st.session_state.esp32_connected = False
        
        if 'esp32_stream' not in st.session_state:
            st.session_state.esp32_stream = None
        
        # Display connection status
        if st.session_state.esp32_connected:
            st.success("🟢 Connected")
        else:
            st.error("🔴 Disconnected")
    
    # ============= CONNECTION CONTROLS =============
    col_connect, col_disconnect = st.columns(2)
    
    with col_connect:
        if st.button("🔌 Connect", type="primary", use_container_width=True):
            try:
                # Try to open video stream
                cap = cv.VideoCapture(stream_url)
                
                # Check if connection successful
                if cap.isOpened():
                    # Test read one frame
                    ret, test_frame = cap.read()
                    
                    if ret:
                        st.session_state.esp32_stream = cap
                        st.session_state.esp32_connected = True
                        st.success(f"✅ Connected to {stream_url}")
                        st.rerun()
                    else:
                        cap.release()
                        st.error("❌ Could not read frames from stream")
                else:
                    st.error("❌ Could not connect to camera")
                    
            except Exception as e:
                st.error(f"❌ Connection error: {str(e)}")
    
    with col_disconnect:
        if st.button("⏹️ Disconnect", use_container_width=True):
            if st.session_state.esp32_stream is not None:
                st.session_state.esp32_stream.release()
                st.session_state.esp32_stream = None
            st.session_state.esp32_connected = False
            st.rerun()
    
    # ============= SECTION 2: LIVE STREAM DISPLAY =============
    if st.session_state.esp32_connected and st.session_state.esp32_stream is not None:
        
        st.markdown("---")
        st.subheader("2️⃣ Live Stream")
        
        # Stream controls
        with st.expander("⚙️ Stream Settings", expanded=False):
            col_set1, col_set2 = st.columns(2)
            
            with col_set1:
                # Frame rate limiter
                fps_limit = st.slider(
                    "Max FPS",
                    min_value=1,
                    max_value=30,
                    value=15,
                    help="Limit frame rate to reduce bandwidth"
                )
                
                # Resolution display toggle
                show_resolution = st.checkbox("Show resolution overlay", value=True)
            
            with col_set2:
                # Quality adjustment (for display only, doesn't affect stream)
                display_scale = st.slider(
                    "Display Scale",
                    min_value=0.25,
                    max_value=2.0,
                    value=1.0,
                    step=0.25,
                    help="Scale factor for display (doesn't affect captured images)"
                )
                
                # Flip options
                flip_horizontal = st.checkbox("Flip horizontal", value=False)
                flip_vertical = st.checkbox("Flip vertical", value=False)
        
        # Create placeholder for video stream
        video_placeholder = st.empty()
        
        # Create placeholder for capture button
        button_placeholder = st.empty()
        
        # Initialize session state for captured frame
        if 'esp32_captured_frame' not in st.session_state:
            st.session_state.esp32_captured_frame = None
        
        # Capture button (placed before stream to avoid rerun issues)
        capture_clicked = button_placeholder.button(
            "📸 Capture Frame",
            type="primary",
            use_container_width=True,
            key="capture_button"
        )
        
        # ============= STREAMING LOOP =============
        # Calculate frame delay based on FPS limit
        frame_delay = 1.0 / fps_limit
        
        try:
            # Read frame from stream
            ret, frame = st.session_state.esp32_stream.read()
            
            if ret:
                # Convert BGR to RGB for display
                frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                
                # Apply transformations
                if flip_horizontal:
                    frame_rgb = cv.flip(frame_rgb, 1)
                if flip_vertical:
                    frame_rgb = cv.flip(frame_rgb, 0)
                
                # Add resolution overlay if enabled
                if show_resolution:
                    h, w = frame_rgb.shape[:2]
                    cv.putText(
                        frame_rgb,
                        f"{w}x{h}",
                        (10, 30),
                        cv.FONT_HERSHEY_SIMPLEX,
                        1,
                        (110, 186, 49),  # THEME['primary'] in RGB
                        2,
                        cv.LINE_AA
                    )
                
                # Handle capture
                if capture_clicked:
                    st.session_state.esp32_captured_frame = frame_rgb.copy()
                    st.toast("📸 Frame captured!", icon="✅")
                
                # Scale for display
                if display_scale != 1.0:
                    h, w = frame_rgb.shape[:2]
                    new_w = int(w * display_scale)
                    new_h = int(h * display_scale)
                    frame_rgb = cv.resize(frame_rgb, (new_w, new_h))
                
                # Display frame
                video_placeholder.image(
                    frame_rgb,
                    channels="RGB",
                    use_container_width=True,
                    caption="Live Stream"
                )
                
                # Auto-refresh for continuous streaming
                time.sleep(frame_delay)
                st.rerun()
                
            else:
                st.error("❌ Lost connection to camera")
                st.session_state.esp32_connected = False
                st.session_state.esp32_stream.release()
                st.session_state.esp32_stream = None
                
        except Exception as e:
            st.error(f"❌ Stream error: {str(e)}")
            st.session_state.esp32_connected = False
            if st.session_state.esp32_stream is not None:
                st.session_state.esp32_stream.release()
                st.session_state.esp32_stream = None
        
        # ============= SECTION 3: CAPTURED FRAME =============
        if st.session_state.esp32_captured_frame is not None:
            st.markdown("---")
            st.subheader("3️⃣ Captured Frame")
            
            col_preview, col_actions = st.columns([2, 1])
            
            with col_preview:
                st.image(
                    st.session_state.esp32_captured_frame,
                    caption="Captured Frame",
                    use_container_width=True
                )
            
            with col_actions:
                st.write("**Export Options:**")
                
                # Export to Tab 1 (as original image)
                if st.button(
                    "📤 Send to Tab 1 (Setup)",
                    use_container_width=True,
                    help="Use this frame as the original image in Tab 1"
                ):
                    # Convert RGB to BGR for OpenCV compatibility
                    frame_bgr = cv.cvtColor(
                        st.session_state.esp32_captured_frame,
                        cv.COLOR_RGB2BGR
                    )
                    st.session_state.original_image = frame_bgr
                    st.success("✅ Frame sent to Tab 1")
                
                # Save as file
                if st.button(
                    "💾 Download Image",
                    use_container_width=True
                ):
                    # Convert to PIL Image for download
                    pil_image = Image.fromarray(st.session_state.esp32_captured_frame)
                    
                    # Create download button
                    import io
                    buf = io.BytesIO()
                    pil_image.save(buf, format="PNG")
                    buf.seek(0)
                    
                    st.download_button(
                        "⬇️ Download PNG",
                        buf,
                        file_name=f"esp32_capture_{int(time.time())}.png",
                        mime="image/png",
                        use_container_width=True
                    )
                
                st.markdown("---")
                
                # Clear captured frame
                if st.button(
                    "🗑️ Clear Capture",
                    use_container_width=True
                ):
                    st.session_state.esp32_captured_frame = None
                    st.rerun()
    
    else:
        # ============= SETUP INSTRUCTIONS =============
        st.info("👆 Connect to your ESP32-CAM to start streaming")
        
        with st.expander("📖 Setup Instructions", expanded=True):
            st.markdown("""
            ### ESP32-CAM Setup Guide
            
            **1. Flash ESP32-CAM Firmware:**
            - Open Arduino IDE
            - Install ESP32 board support
            - Load example: `File → Examples → ESP32 → Camera → CameraWebServer`
            - Select your camera model (e.g., `AI Thinker`)
            - Upload to ESP32-CAM
            
            **2. Configure WiFi:**
            - Edit `ssid` and `password` in the code
            - Set your network credentials
            - Upload again
            
            **3. Find IP Address:**
            - Open Serial Monitor (115200 baud)
            - After boot, ESP32 will print its IP address
            - Example: `Camera Ready! Use 'http://192.168.1.100' to connect`
            
            **4. Test Stream:**
            - Open browser and go to `http://ESP32-IP`
            - Click "Start Stream" button
            - Verify video appears
            - Note: Stream URL is usually `http://ESP32-IP:81/stream`
            
            **5. Connect Here:**
            - Enter the IP address above
            - Keep port as `81` (standard)
            - Click "Connect"
            
            ---
            
            ### Troubleshooting
            
            **Cannot connect:**
            - ✅ Check ESP32 is powered on
            - ✅ Verify IP address is correct
            - ✅ Ensure both devices on same network
            - ✅ Try pinging ESP32: `ping 192.168.x.x`
            
            **Stream is slow/laggy:**
            - 🔧 Reduce "Max FPS" in stream settings
            - 🔧 Lower resolution in ESP32 web interface
            - 🔧 Reduce JPEG quality in ESP32 settings
            - 🔧 Check WiFi signal strength
            
            **Connection drops:**
            - ⚡ Check ESP32 power supply (needs stable 5V)
            - ⚡ Reduce frame rate
            - ⚡ Shorten distance to WiFi router
            
            ---
            
            ### Network Security Note
            
            ⚠️ **Important:** ESP32-CAM streams are **unencrypted HTTP**. 
            Only use on trusted local networks. Do not expose to the internet.
            """)
        
        # Quick test section
        with st.expander("🧪 Quick Connection Test", expanded=False):
            st.write("**Test if ESP32 is reachable:**")
            
            if st.button("🔍 Ping ESP32", use_container_width=True):
                import subprocess
                import platform
                
                try:
                    # Determine ping command based on OS
                    param = '-n' if platform.system().lower() == 'windows' else '-c'
                    
                    # Execute ping
                    result = subprocess.run(
                        ['ping', param, '1', esp32_ip],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    
                    if result.returncode == 0:
                        st.success(f"✅ {esp32_ip} is reachable!")
                    else:
                        st.error(f"❌ {esp32_ip} is not reachable")
                        st.code(result.stdout)
                        
                except subprocess.TimeoutExpired:
                    st.error("❌ Ping timeout - device not responding")
                except Exception as e:
                    st.error(f"❌ Ping failed: {str(e)}")
