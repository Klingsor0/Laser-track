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
optical_experiment_project/           # Root project directory
│
├── README.md
├── requirements.txt                  # Shared dependencies
├── .gitignore
│
├── shared_utils/                     # ✅ SHARED BY ALL VERSIONS
│   ├── __init__.py
│   ├── image_processing.py          # load_img, crop_img, rotation, blur
│   ├── canvas_utils.py              # canvas_to_pts, polygon_area
│   ├── curve_analysis.py            # spline fitting, critical points
│   ├── plotting.py                  # themed Plotly plots
│   ├── optical_experiment.py        # CaptureSession, ExperimentData
│   ├── straight_line_snake.py       # straight-line constrained snake
│   ├── angle_extraction.py          # complete angle extraction pipeline
│   ├── preprocessing.py             # ⚠️ TO CREATE (from tab2)
│   ├── calibration.py               # ⚠️ TO CREATE (from tab1)
│   ├── snake1.py                    # YOUR original snake
│   ├── models.py                    # YOUR model classes
│   └── plots.py                     # YOUR matplotlib plots
│
├── streamlit_app/                   # 🔵 CURRENT WORKING VERSION
│   ├── app.py                       # Main entry point
│   ├── config/
│   │   ├── __init__.py
│   │   └── theme.py                 # Streamlit CSS/colors
│   ├── tabs/
│   │   ├── __init__.py
│   │   ├── tab1_setup.py           # Upload, calibration, ROI
│   │   ├── tab2_preprocessing.py   # Filters (blur, contrast, morph)
│   │   ├── tab3_edge_detection.py  # Manual curve + snake
│   │   ├── tab4_analysis.py        # Spline fitting, models
│   │   ├── tab5_validation.py      # Experimental validation
│   │   ├── tab6_esp32_camera.py    # Camera connection
│   │   ├── tab7_line_tracking.py   # Hough line detection
│   │   ├── tab9_batch_acquisition.py # Capture N frames
│   │   └── tab10_angle_analysis.py   # Statistics, histograms
│   ├── README.md
│   └── requirements_streamlit.txt
│
├── dash_app/                        # 🟢 DASH VERSION (separate branch)
│   ├── app.py
│   ├── layouts/
│   ├── callbacks/
│   └── requirements_dash.txt
│
├── pyqt_app/                        # 🔴 PYQT6 VERSION (TO CREATE)
│   ├── main.py                      # PyQt6 entry point
│   ├── backend/                     # Business logic (GUI-independent)
│   │   ├── __init__.py
│   │   ├── camera_manager.py       # ⚠️ TO CREATE (from tab6)
│   │   ├── acquisition_controller.py # ⚠️ TO CREATE (from tab9)
│   │   ├── analysis_controller.py    # ⚠️ TO CREATE (from tab10)
│   │   └── curve_fitting_controller.py # ⚠️ TO CREATE (from tab4)
│   ├── gui/                         # PyQt6 widgets
│   │   ├── __init__.py
│   │   ├── main_window.py
│   │   ├── camera_widget.py
│   │   ├── acquisition_widget.py
│   │   └── analysis_widget.py
│   ├── resources/
│   ├── tests/
│   └── requirements_pyqt.txt
│
├── tests/                           # Shared tests
│   ├── test_utils.py
│   └── test_data/
│
├── docs/                            # Documentation
│   ├── CLAUDE_CODE_HANDOFF.md
│   ├── PROJECT_STRUCTURE.md
│   ├── QUICK_START.md
│   └── LINE_TRACKING_GUIDE.md
│
└── data/                            # Data storage
    ├── sessions/
    └── exports/
```

---

## Key Shared Components

### **1. ROI Selection** (Both workflows)
- **Tab 1:** Polygon drawing for complex shapes
- **Tab 9:** Rectangle for laser line region
- **Shared in:** `shared_utils/canvas_utils.py` processes canvas data
- **For CLI:** Use `cv.selectROI()` or matplotlib

**Import in Streamlit:**
```python
# streamlit_app/tabs/tab1_setup.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'shared_utils'))
from canvas_utils import canvas_to_pts
```

**Import in PyQt6:**
```python
# pyqt_app/backend/acquisition_controller.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'shared_utils'))
from canvas_utils import canvas_to_pts
```

### **2. Image Filtering** (Both workflows)
- **Tab 2:** Gaussian blur, contrast, morphology
- **Shared in:** `shared_utils/image_processing.py` has blur only
- **Missing:** Contrast adjustment, morphology → extract to `shared_utils/preprocessing.py`

### **3. Snake Algorithm** (CORE - both workflows!)
- **Your `shared_utils/snake1.py`:** Flexible snake with adjustable rigidity
- **Workflow A:** Low rigidity (β~0.1) for curves
- **Workflow B:** High rigidity (β~10) for straight lines
- **My addition:** `shared_utils/straight_line_snake.py` wraps yours with presets

### **4. Edge Detection** (Both workflows)
- **Tab 2, Tab 7, Tab 9:** All have Canny edge detection
- **Current:** Duplicated in multiple places
- **Should be:** Single function in `shared_utils/image_processing.py`

### **5. Calibration** (Workflow A only)
- **Tab 1:** Draw line, set physical length, calculate scale
- **Stored:** `session_state.calibration` dict
- **Should extract:** `shared_utils/calibration.py`

---

## What to Extract from Tabs to Shared Utils

### **From Tab 2 (Preprocessing):**

**Create:** `shared_utils/preprocessing.py`

```python
# shared_utils/preprocessing.py

import cv2 as cv
import numpy as np

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

**Create:** `shared_utils/calibration.py`

```python
# shared_utils/calibration.py

import numpy as np

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

### **Add to `shared_utils/image_processing.py`:**

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

**Goal:** Each utility in `shared_utils/` can be tested independently

**Prerequisites:** Restructure project first (see PROJECT_STRUCTURE.md)
- Move `streamlit_app/utils/` → `shared_utils/` at root
- Update imports in streamlit_app
- Verify Streamlit still works

**Add test blocks to:**
1. `shared_utils/image_processing.py`
2. `shared_utils/curve_analysis.py`
3. `shared_utils/optical_experiment.py`
4. Create `shared_utils/preprocessing.py` (extract from tab2)
5. Create `shared_utils/calibration.py` (extract from tab1)

**Pattern:**
```python
# In each shared_utils/*.py file
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='Input file')
    args = parser.parse_args()
    
    # Test the functions in this module
    # Print results
    # Save outputs
```

**Test:** `python shared_utils/image_processing.py test.png`

### **Phase 2: Create Backend Controllers**

**Goal:** Business logic separate from UI in `pyqt_app/backend/`

**Extract from streamlit tabs:**
- `streamlit_app/tabs/tab6_esp32_camera.py` → `pyqt_app/backend/camera_manager.py`
- `streamlit_app/tabs/tab9_batch_acquisition.py` → `pyqt_app/backend/acquisition_controller.py`
- `streamlit_app/tabs/tab10_angle_analysis.py` → `pyqt_app/backend/analysis_controller.py`
- `streamlit_app/tabs/tab4_analysis.py` → `pyqt_app/backend/curve_fitting_controller.py`

**Pattern:**
```python
# pyqt_app/backend/acquisition_controller.py

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'shared_utils'))

from optical_experiment import CaptureSession
from angle_extraction import extract_angle_from_frame

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

**Start with Tab 6 → `pyqt_app/gui/camera_widget.py`:**
```python
# pyqt_app/gui/camera_widget.py

from PyQt6.QtWidgets import QWidget, QLabel, QPushButton, QVBoxLayout
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QImage, QPixmap
import cv2 as cv
import sys
from pathlib import Path

# Add shared_utils to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'shared_utils'))

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

### **Step 1: Add Test to `shared_utils/image_processing.py`**

**Prerequisites:** Project must be restructured (see PROJECT_STRUCTURE.md Phase 1)

```python
# Add at end of shared_utils/image_processing.py

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

**Test:** 
```bash
cd optical_experiment_project
python shared_utils/image_processing.py test.png --blur 7
```

**Report:** Terminal output + confirm output.png created

### **Step 2: Create `shared_utils/preprocessing.py`**

Extract contrast/morphology from tab2 into new utils file with test block.

**Test:** `python shared_utils/preprocessing.py test.png`

### **Step 3: Create `shared_utils/calibration.py`**

Extract calibration logic from tab1 into new utils file with test block.

**Test:** `python shared_utils/calibration.py --test`

### **Step 4: Add edge detection to `shared_utils/image_processing.py`**

Move Canny edge detection to shared utility.

### **Step 5: Test All Utils Standalone**

Verify every shared_utils/*.py can run with test data.

### **Step 6: Create First Backend Controller**

Extract camera logic from streamlit_app/tabs/tab6 → `pyqt_app/backend/camera_manager.py`

Test without GUI.

### **Step 7: Create PyQt6 Camera Widget**

Simple window with live preview in `pyqt_app/gui/camera_widget.py`

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

**IMPORTANT: Before any coding, read PROJECT_STRUCTURE.md**

### **Step 0: Restructure Project** (One-time setup)

**Current state:**
```
optical_experiment_project/
└── streamlit_app/
    ├── utils/  ← Need to move!
    ├── tabs/
    └── config/
```

**Target state:**
```
optical_experiment_project/
├── shared_utils/  ← Move utils here!
├── streamlit_app/
│   ├── tabs/
│   └── config/
└── pyqt_app/     ← Create later
```

**Actions:**
1. Move `streamlit_app/utils/` to project root as `shared_utils/`
2. Update all imports in `streamlit_app/` to use shared_utils
3. Test that Streamlit still works

**After restructure succeeds, move to Step 1.**

### **Step 1: Add standalone test to `shared_utils/image_processing.py`**

See detailed code in "Modular Steps" section above.

Run: `python shared_utils/image_processing.py test.png`

Report back with:
1. Terminal output
2. Confirmation output.png exists
3. Any errors encountered

**Do NOT proceed to Step 2 until Step 1 works!**

---

## File Locations

**Templates/Examples:** `/mnt/user-data/outputs/`
(These are reference files created during our conversation)

**Your actual project structure should be:**
```
optical_experiment_project/
├── shared_utils/         ← Core utilities (framework-agnostic)
├── streamlit_app/        ← Working Streamlit version
├── dash_app/            ← Dash version (separate branch)
└── pyqt_app/            ← PyQt6 version (to create)
```

**Your existing files (already in streamlit_app):**
- `streamlit_app/utils/snake1.py` → move to `shared_utils/snake1.py`
- `streamlit_app/utils/models.py` → move to `shared_utils/models.py`
- `streamlit_app/utils/plots.py` → move to `shared_utils/plots.py`

**Files from /mnt/user-data/outputs/ to integrate:**
- Copy utility files to `shared_utils/`
- Keep tab files in `streamlit_app/tabs/`
- Use docs/ files as reference
