# Claude Code Handoff Document - COMPLETE

## Project Overview

**Streamlit Image Processing Application with TWO Main Workflows:**

### **Workflow A: Profile Characterization (Tabs 1-5)** ✅ Existing
Upload static image → ROI selection → Rotation → Calibration → Edge detection → Manual snake drawing → Curve fitting → Model comparison → Export

### **Workflow B: Laser Angle Measurement (Tabs 6-10)** 🆕 New
Connect camera → Configure preprocessing → Capture N frames at distance → Auto angle extraction → Statistics → Export

**BOTH workflows share common utilities!**

---

## Complete File Structure

```
utils/ (Framework-agnostic - KEEP THESE)
├── image_processing.py      ✅ load_img, crop_img, rotation, blur
├── canvas_utils.py          ✅ canvas_to_pts, polygon_area
├── curve_analysis.py        ✅ spline fitting, critical points
├── plotting.py              ✅ themed Plotly plots
├── optical_experiment.py    ✅ CaptureSession, ExperimentData
├── straight_line_snake.py   ✅ straight-line constrained snake
├── angle_extraction.py      ✅ complete angle extraction pipeline
├── snake1.py                ⚠️ YOUR original (not in my files)
├── models.py                ⚠️ YOUR models (not in my files)
└── plots.py                 ⚠️ YOUR plots (not in my files)

tabs/ (Streamlit-dependent - REFACTOR THESE)
├── tab1_setup.py            ❌ Upload, calibration, ROI
├── tab2_preprocessing.py    ❌ Filters (blur, contrast, morph)
├── tab3_edge_detection.py   ❌ Manual curve + snake
├── tab4_analysis.py         ❌ Spline fitting, models
├── tab5_validation.py       ❌ Experimental validation
├── tab6_esp32_camera.py     ❌ Camera connection
├── tab7_line_tracking.py    ❌ Hough line detection
├── tab9_batch_acquisition.py ❌ Capture N frames
└── tab10_angle_analysis.py  ❌ Statistics, histograms

config/
└── theme.py                 ✅ Colors, CSS (keep)
```

---

## Key Shared Components

### **1. ROI Selection** (Both workflows)
- **Tab 1:** Polygon drawing for complex shapes
- **Tab 9:** Rectangle for laser line region
- **Utils:** `canvas_utils.py` processes canvas data
- **For CLI:** Use `cv.selectROI()` or matplotlib

### **2. Image Filtering** (Both workflows)
- **Tab 2:** Gaussian blur, contrast, morphology
- **Utils:** `image_processing.py` has blur only
- **Missing:** Contrast adjustment, morphology → extract from tab2

### **3. Snake Algorithm** (CORE - both workflows!)
- **Your `snake1.py`:** Flexible snake with adjustable rigidity
- **Workflow A:** Low rigidity (β~0.1) for curves
- **Workflow B:** High rigidity (β~10) for straight lines
- **My addition:** `straight_line_snake.py` wraps yours with presets

### **4. Edge Detection** (Both workflows)
- **Tab 2, Tab 7, Tab 9:** All have Canny edge detection
- **Current:** Duplicated in multiple places
- **Should be:** Single function in `utils/image_processing.py`

### **5. Calibration** (Workflow A only)
- **Tab 1:** Draw line, set physical length, calculate scale
- **Stored:** `session_state.calibration` dict
- **Should extract:** `utils/calibration.py`

---

## What to Extract from Tabs to Utils

### **From Tab 2 (Preprocessing):**

```python
# utils/preprocessing.py (NEW)

def adjust_contrast(img, alpha=1.0, beta=0):
    """Alpha: contrast multiplier, Beta: brightness offset"""
    return cv.convertScaleAbs(img, alpha=alpha, beta=beta)

def morphological_operation(img, operation, kernel_size=5):
    """Apply erosion, dilation, opening, or closing"""
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (kernel_size, kernel_size))
    ops = {
        'erosion': lambda i, k: cv.erode(i, k),
        'dilation': lambda i, k: cv.dilate(i, k),
        'opening': lambda i, k: cv.morphologyEx(i, cv.MORPH_OPEN, k),
        'closing': lambda i, k: cv.morphologyEx(i, cv.MORPH_CLOSE, k)
    }
    return ops[operation](img, kernel)

if __name__ == '__main__':
    # Standalone test
    import sys
    img = cv.imread(sys.argv[1])
    adjusted = adjust_contrast(img, 1.5, 10)
    cv.imwrite('test_contrast.png', adjusted)
    print("✅ Contrast adjusted")
```

### **From Tab 1 (Calibration):**

```python
# utils/calibration.py (NEW)

def calculate_scale_from_line_endpoints(pt1, pt2, real_distance, unit='mm'):
    """
    Calculate scale factor from line endpoints
    
    Args:
        pt1, pt2: (x, y) tuples
        real_distance: Known physical length
        unit: Physical unit
    
    Returns:
        dict with scale, unit, pixel_distance
    """
    pixel_dist = np.sqrt((pt2[0]-pt1[0])**2 + (pt2[1]-pt1[1])**2)
    scale = real_distance / pixel_dist
    return {
        'scale': scale,
        'unit': unit,
        'reference_value': real_distance,
        'pixel_distance': pixel_dist
    }

def apply_calibration(pixel_value, calibration):
    """Convert pixel measurement to physical units"""
    return pixel_value * calibration['scale']

if __name__ == '__main__':
    # Test
    scale_data = calculate_scale_from_line_endpoints((0, 0), (100, 0), 10.0, 'mm')
    print(f"Scale: {scale_data['scale']} mm/pixel")
```

### **Add to `image_processing.py`:**

```python
def detect_edges_canny(img, low_threshold=50, high_threshold=150, 
                      blur_kernel=3, apply_morphology=False):
    """
    Canny edge detection with optional preprocessing
    
    Args:
        img: Grayscale image
        low_threshold: Canny low threshold
        high_threshold: Canny high threshold
        blur_kernel: Gaussian blur before edge detection (0 = no blur)
        apply_morphology: Clean edges with morphological closing
    
    Returns:
        Binary edge image
    """
    if blur_kernel > 0:
        img = gaussian_blur(img, blur_kernel)
    
    edges = cv.Canny(img, low_threshold, high_threshold)
    
    if apply_morphology:
        kernel = np.ones((3, 3), np.uint8)
        edges = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel)
    
    return edges
```

---

## Video Processing Optimization

**Current problem:** Streamlit reruns on every frame → slow

**Solutions:**

### **1. Frame Buffering (Easiest)**
```python
# Capture all frames FIRST (fast)
frames = []
for i in range(60):
    ret, frame = cap.read()
    frames.append(frame)

# Then process (no camera delays)
for frame in frames:
    angle = extract_angle_from_frame(frame)
```

### **2. Reduce Resolution**
```python
cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
# Smaller images = faster processing
```

### **3. Skip Preview Frames**
```python
for i in range(60):
    frame = cap.read()
    angle = extract_angle(frame)
    
    if i % 5 == 0:  # Preview every 5th frame
        display(frame)
```

### **4. Threading (PyQt6 solution)**
```python
class CaptureThread(QThread):
    frame_ready = Signal(np.ndarray, float)
    
    def run(self):
        for i in range(self.num_frames):
            frame = self.cap.read()
            angle = extract_angle(frame)
            self.frame_ready.emit(frame, angle)
```

---

## Refactoring Strategy

### **Phase 1: Make Utils Standalone** 🎯 START HERE

**Goal:** Each utility can be tested independently

**Add test blocks to:**
1. `utils/image_processing.py`
2. `utils/curve_analysis.py`
3. `utils/optical_experiment.py`
4. Create `utils/preprocessing.py`
5. Create `utils/calibration.py`

**Pattern:**
```python
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='Input file')
    args = parser.parse_args()
    
    # Test the functions in this module
    # Print results
    # Save outputs
```

**Test:** `python utils/image_processing.py test.png`

### **Phase 2: Create Backend Controllers**

**Goal:** Business logic separate from UI

**Extract from tabs:**
- Tab 6 → `backend/camera_manager.py`
- Tab 9 → `backend/acquisition_controller.py`
- Tab 10 → `backend/analysis_controller.py`
- Tab 4 → `backend/curve_fitting_controller.py`

**Pattern:**
```python
# backend/acquisition_controller.py

class AcquisitionController:
    def __init__(self, camera):
        self.camera = camera
        self.session = None
    
    def start_capture(self, distance, num_frames, preprocessing_config, callbacks):
        self.session = CaptureSession(distance, 'mm')
        
        for i in range(num_frames):
            frame = self.camera.read()
            angle = extract_angle_from_frame(frame, **preprocessing_config)
            self.session.add_frame(frame, angle)
            
            # Callback for UI updates (no GUI imports here!)
            if callbacks:
                callbacks.on_progress(i+1, num_frames)
                callbacks.on_frame(frame, angle)
        
        return self.session
```

### **Phase 3: PyQt6 Migration** (Incremental)

**Order:**
1. Camera preview (simplest)
2. Acquisition (uses camera)
3. Analysis (uses saved data)
4. Profile characterization (more complex)

**Start with Tab 6:**
```python
# gui_pyqt/camera_widget.py

from PyQt6.QtWidgets import QWidget, QLabel, QPushButton, QVBoxLayout
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QImage, QPixmap
import cv2 as cv

class CameraWidget(QWidget):
    def __init__(self, camera_id=0):
        super().__init__()
        
        # UI elements
        self.image_label = QLabel()
        self.connect_btn = QPushButton("Connect")
        self.disconnect_btn = QPushButton("Disconnect")
        
        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.connect_btn)
        layout.addWidget(self.disconnect_btn)
        self.setLayout(layout)
        
        # Camera
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        
        # Connections
        self.connect_btn.clicked.connect(self.on_connect)
        self.disconnect_btn.clicked.connect(self.on_disconnect)
    
    def on_connect(self):
        self.cap = cv.VideoCapture(0)
        self.timer.start(33)  # 30 FPS
    
    def on_disconnect(self):
        self.timer.stop()
        if self.cap:
            self.cap.release()
    
    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qimg = QImage(rgb.data, w, h, w*ch, QImage.Format.Format_RGB888)
            self.image_label.setPixmap(QPixmap.fromImage(qimg))
```

---

## Modular Steps for Claude Code

### **Step 1: Add Test to `image_processing.py`**

```python
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test image processing')
    parser.add_argument('image', help='Input image path')
    parser.add_argument('--blur', type=int, default=5)
    parser.add_argument('--output', default='output.png')
    args = parser.parse_args()
    
    # Load with OpenCV
    img = cv.imread(args.image)
    if img is None:
        print(f"❌ Could not load {args.image}")
        exit(1)
    
    print(f"✅ Loaded: {img.shape}")
    
    # Test ROI crop
    roi = crop_img(img, [[100, 100], [300, 100], [300, 300], [100, 300]])
    print(f"✅ ROI cropped: {roi.shape}")
    
    # Test blur
    blurred = gaussian_blur(roi, args.blur)
    cv.imwrite(args.output, blurred)
    print(f"✅ Saved: {args.output}")
```

**Test:** `python utils/image_processing.py test.png --blur 7`

**Report:** Terminal output + confirm output.png created

### **Step 2: Create `preprocessing.py`**

Extract contrast/morphology from tab2 into new utils file with test block.

**Test:** `python utils/preprocessing.py test.png`

### **Step 3: Create `calibration.py`**

Extract calibration logic from tab1 into new utils file with test block.

**Test:** `python utils/calibration.py --test`

### **Step 4: Add edge detection to `image_processing.py`**

Move Canny edge detection to shared utility.

### **Step 5: Test All Utils Standalone**

Verify every utils/*.py can run with test data.

### **Step 6: Create First Backend Controller**

Extract camera logic from tab6 → `backend/camera_manager.py`

Test without GUI.

### **Step 7: Create PyQt6 Camera Widget**

Simple window with live preview.

### **Step 8+: Continue Incrementally**

One controller → one widget → test → next

---

## Key Principles

### ✅ DO:
- Test each utility standalone before moving on
- One file at a time (small, testable changes)
- Keep utils/ GUI-independent
- Use callbacks for UI updates in backend
- Git commit after each working step

### ❌ DON'T:
- Modify multiple files at once
- Put GUI code in utils/
- Skip standalone testing
- Delete Streamlit version (keep as reference)
- Rush ahead without verifying previous step

---

## Success Criteria

**Phase 1 Complete:**
- [ ] All utils/*.py have `if __name__` blocks
- [ ] Can test each utility independently
- [ ] Preprocessing extracted from tabs
- [ ] Calibration extracted from tabs

**Phase 2 Complete:**
- [ ] Backend controllers exist
- [ ] Controllers run without GUI
- [ ] Test scripts verify controller logic

**Phase 3 Complete:**
- [ ] PyQt6 camera preview works
- [ ] PyQt6 acquisition works
- [ ] PyQt6 analysis works
- [ ] Feature parity with Streamlit

---

## Immediate First Task

**Add standalone test to `utils/image_processing.py`**

See Step 1 above for exact code.

Run: `python utils/image_processing.py test.png`

Report back with:
1. Terminal output
2. Confirmation output.png exists
3. Any errors encountered

**Do NOT proceed to Step 2 until Step 1 works!**

---

## File Locations

All project files: `/mnt/user-data/outputs/`

Your existing files (not in outputs):
- `utils/snake1.py`
- `utils/models.py`
- `utils/plots.py`

Copy outputs to your project:
```bash
cp -r /mnt/user-data/outputs/* ~/my_project/
```
