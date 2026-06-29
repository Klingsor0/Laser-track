# Project Structure - Complete Layout

## Overview

**Project has THREE parallel implementations:**
1. **Streamlit** - Current working version (prototype/development)
2. **Dash** - Alternative web framework (on separate branch)
3. **PyQt6** - Desktop application (migration target)

**All three share the same core utilities!**

---

## Complete Directory Structure

```
optical_experiment_project/           # Root project directory
│
├── README.md                         # Main project documentation
├── requirements.txt                  # Shared dependencies
├── .gitignore
│
├── shared_utils/                     # ✅ SHARED BY ALL VERSIONS
│   ├── __init__.py
│   │
│   ├── image_processing.py          # Image load, crop, rotate, blur
│   ├── canvas_utils.py              # Polygon processing
│   ├── preprocessing.py             # Contrast, morphology (to extract)
│   ├── calibration.py               # Scale calculation (to extract)
│   │
│   ├── curve_analysis.py            # Spline fitting, critical points
│   ├── plotting.py                  # Themed plots (Plotly)
│   │
│   ├── optical_experiment.py        # CaptureSession, ExperimentData
│   ├── straight_line_snake.py       # Straight-line snake wrapper
│   ├── angle_extraction.py          # Complete angle pipeline
│   │
│   ├── snake1.py                    # YOUR original snake
│   ├── models.py                    # YOUR model classes
│   └── plots.py                     # YOUR matplotlib plots
│
├── streamlit_app/                   # 🔵 STREAMLIT VERSION
│   ├── app.py                       # Main entry point
│   │
│   ├── config/
│   │   ├── __init__.py
│   │   └── theme.py                 # Streamlit CSS/colors
│   │
│   ├── tabs/                        # Streamlit tab modules
│   │   ├── __init__.py
│   │   ├── tab1_setup.py           # Upload, calibration, ROI
│   │   ├── tab2_preprocessing.py   # Filters
│   │   ├── tab3_edge_detection.py  # Manual snake
│   │   ├── tab4_analysis.py        # Curve fitting
│   │   ├── tab5_validation.py      # Validation
│   │   ├── tab6_esp32_camera.py    # Camera connection
│   │   ├── tab7_line_tracking.py   # Hough lines
│   │   ├── tab9_batch_acquisition.py  # Angle capture
│   │   └── tab10_angle_analysis.py    # Statistics
│   │
│   ├── README.md                    # Streamlit-specific docs
│   └── requirements_streamlit.txt   # Streamlit dependencies
│
├── dash_app/                        # 🟢 DASH VERSION (separate branch)
│   ├── app.py
│   ├── layouts/
│   ├── callbacks/
│   ├── README.md
│   └── requirements_dash.txt
│
├── pyqt_app/                        # 🔴 PYQT6 VERSION (new)
│   ├── main.py                      # PyQt6 entry point
│   │
│   ├── backend/                     # ✅ Business logic (GUI-independent)
│   │   ├── __init__.py
│   │   ├── camera_manager.py       # Camera operations
│   │   ├── acquisition_controller.py  # Batch capture logic
│   │   ├── analysis_controller.py     # Statistics/plots logic
│   │   └── curve_fitting_controller.py  # Profile workflow logic
│   │
│   ├── gui/                         # ❌ PyQt6 GUI components
│   │   ├── __init__.py
│   │   ├── main_window.py          # Main QMainWindow
│   │   ├── camera_widget.py        # Camera preview
│   │   ├── acquisition_widget.py   # Batch capture UI
│   │   ├── analysis_widget.py      # Plots/stats UI
│   │   ├── profile_widget.py       # Curve analysis UI
│   │   └── dialogs/                # Dialog windows
│   │       ├── roi_dialog.py
│   │       └── calibration_dialog.py
│   │
│   ├── resources/                   # Icons, styles
│   │   ├── icons/
│   │   └── styles.qss              # Qt stylesheet
│   │
│   ├── tests/                       # Unit tests
│   │   ├── test_backend.py
│   │   └── test_controllers.py
│   │
│   ├── README.md                    # PyQt6-specific docs
│   └── requirements_pyqt.txt        # PyQt6 dependencies
│
├── tests/                           # 🧪 SHARED TESTS
│   ├── __init__.py
│   ├── test_utils.py               # Test shared utilities
│   ├── test_image_processing.py
│   ├── test_angle_extraction.py
│   └── test_data/                  # Sample images/data
│       ├── test_image.png
│       └── test_laser_line.png
│
├── docs/                            # 📚 DOCUMENTATION
│   ├── CLAUDE_CODE_HANDOFF.md      # Migration guide
│   ├── QUICK_START.md              # Quick reference
│   ├── LINE_TRACKING_GUIDE.md      # Line detection docs
│   ├── ESP32_SETUP.md              # Hardware setup
│   └── API_REFERENCE.md            # Utils API docs
│
├── data/                            # 📁 DATA STORAGE
│   ├── sessions/                   # Saved capture sessions (JSON)
│   ├── exports/                    # Exported results (CSV, plots)
│   └── calibrations/               # Saved calibration configs
│
└── scripts/                         # 🛠️ UTILITY SCRIPTS
    ├── migrate_streamlit_to_pyqt.py  # Migration helper
    ├── test_all_utils.py             # Test runner
    └── generate_test_data.py         # Create test images
```

---

## Key Principles

### **1. Shared Utils = Single Source of Truth**

```python
# In streamlit_app/tabs/tab9_batch_acquisition.py
import sys
sys.path.insert(0, '../shared_utils')  # Add parent shared_utils
from angle_extraction import extract_angle_from_frame

# In pyqt_app/backend/acquisition_controller.py
import sys
sys.path.insert(0, '../../shared_utils')  # Add shared_utils
from angle_extraction import extract_angle_from_frame

# Same utility, different UIs!
```

**Alternative (better):** Install shared_utils as package:
```bash
cd optical_experiment_project
pip install -e ./shared_utils
```

Then all apps just:
```python
from shared_utils.angle_extraction import extract_angle_from_frame
```

### **2. Version Isolation**

Each app directory is **self-contained**:
- Has its own `app.py` / `main.py`
- Has its own `requirements_*.txt`
- Can run independently
- Shares only `shared_utils/`

### **3. PyQt6 Backend/GUI Separation**

```
pyqt_app/
├── backend/          ← No PyQt6 imports! Pure logic.
│   └── *.py          ← Can be tested from terminal
└── gui/              ← PyQt6 widgets only
    └── *.py          ← Connects backend to UI via signals
```

---

## Import Strategy

### **Option A: Path Manipulation (Quick)**

```python
# At top of any file needing shared_utils
import sys
from pathlib import Path

# Add shared_utils to path
project_root = Path(__file__).parent.parent  # Go up to project root
sys.path.insert(0, str(project_root / 'shared_utils'))

# Now import normally
from angle_extraction import extract_angle_from_frame
```

### **Option B: Package Installation (Professional)**

**1. Make shared_utils a package:**

Create `shared_utils/setup.py`:
```python
from setuptools import setup, find_packages

setup(
    name='optical_utils',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'opencv-python',
        'scipy',
    ]
)
```

**2. Install in development mode:**
```bash
cd optical_experiment_project
pip install -e ./shared_utils
```

**3. Import from anywhere:**
```python
# In any file in any app
from optical_utils.angle_extraction import extract_angle_from_frame
from optical_utils.image_processing import gaussian_blur
```

**Benefits:**
- Clean imports
- No path manipulation
- Works in any directory
- Professional structure

---

## Workflow-Specific Files

### **Streamlit App Structure**

```
streamlit_app/
├── app.py                    # Tabs 1-10 all in one
├── tabs/
│   └── tab*.py              # Each tab is a module with render()
└── config/
    └── theme.py             # Streamlit-specific styling
```

**Run:** `cd streamlit_app && streamlit run app.py`

### **PyQt6 App Structure**

```
pyqt_app/
├── main.py                   # QApplication entry point
├── backend/                  # Controllers (no GUI)
│   ├── camera_manager.py
│   ├── acquisition_controller.py
│   └── analysis_controller.py
└── gui/                      # Widgets (GUI only)
    ├── main_window.py        # Tabs/dock widgets
    ├── camera_widget.py
    └── acquisition_widget.py
```

**Run:** `cd pyqt_app && python main.py`

---

## Migration Strategy

### **Phase 1: Restructure Current Project** 🎯 DO THIS FIRST

```bash
# Current state (everything in streamlit_app/)
optical_experiment_project/
└── streamlit_app/
    ├── app.py
    ├── utils/          ← Needs to move!
    ├── tabs/
    └── config/

# Target state
optical_experiment_project/
├── shared_utils/       ← Move utils here!
├── streamlit_app/
│   ├── app.py
│   ├── tabs/
│   └── config/
└── pyqt_app/          ← Create new
```

**Steps:**

1. **Create new directories:**
   ```bash
   cd optical_experiment_project
   mkdir shared_utils
   mkdir pyqt_app
   ```

2. **Move utils:**
   ```bash
   mv streamlit_app/utils/* shared_utils/
   rmdir streamlit_app/utils
   ```

3. **Update imports in streamlit_app:**
   ```python
   # Old: from utils.angle_extraction import ...
   # New: 
   import sys
   from pathlib import Path
   sys.path.insert(0, str(Path(__file__).parent.parent / 'shared_utils'))
   from angle_extraction import extract_angle_from_frame
   ```

4. **Test Streamlit still works:**
   ```bash
   cd streamlit_app
   streamlit run app.py
   ```

### **Phase 2: Create PyQt6 Skeleton**

```bash
cd pyqt_app
mkdir backend gui resources tests
touch main.py
touch backend/__init__.py
touch gui/__init__.py
```

Create minimal `main.py`:
```python
import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Optical Experiment - PyQt6")
        self.setCentralWidget(QLabel("Hello PyQt6!"))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
```

Test: `python main.py`

### **Phase 3: Add Utils Testing**

Each shared_utils/*.py gets test block (as discussed).

### **Phase 4: Extract Backend from Streamlit**

Create `pyqt_app/backend/` controllers from streamlit tabs.

### **Phase 5: Build PyQt6 GUI**

Connect backend to Qt widgets.

---

## Git Branch Strategy

```bash
# Main branch: Streamlit (stable)
main

# Development branches
├── dash-version              # Dash implementation
├── pyqt6-migration          # PyQt6 work
└── feature/shared-utils     # Restructuring

# Workflow
git checkout -b pyqt6-migration
# Work on PyQt6
git commit -m "Step 1: Created PyQt6 skeleton"
# When stable, merge to main
```

---

## Requirements Files

### **Root `requirements.txt` (Shared)**
```txt
# Core dependencies used by ALL versions
numpy>=1.24.0
opencv-python>=4.8.0
scipy>=1.11.0
pandas>=2.0.0
pillow>=10.0.0
matplotlib>=3.7.0
```

### **`streamlit_app/requirements_streamlit.txt`**
```txt
# Streamlit-specific
-r ../requirements.txt  # Include shared
streamlit>=1.28.0
plotly>=5.17.0
streamlit-drawable-canvas>=0.9.0
```

### **`pyqt_app/requirements_pyqt.txt`**
```txt
# PyQt6-specific
-r ../requirements.txt  # Include shared
PyQt6>=6.6.0
pyqtgraph>=0.13.0  # For fast plotting
```

**Install:**
```bash
# For Streamlit
pip install -r streamlit_app/requirements_streamlit.txt

# For PyQt6
pip install -r pyqt_app/requirements_pyqt.txt
```

---

## Current File Locations (What You Have Now)

**Your existing structure:**
```
streamlit_app/
├── app.py
├── utils/
│   ├── image_processing.py
│   ├── canvas_utils.py
│   ├── curve_analysis.py
│   ├── plotting.py
│   ├── optical_experiment.py
│   ├── straight_line_snake.py
│   ├── angle_extraction.py
│   ├── snake1.py          ← YOUR file
│   ├── models.py          ← YOUR file
│   └── plots.py           ← YOUR file
├── tabs/
│   ├── tab1_setup.py
│   ├── tab2_preprocessing.py
│   ├── ...
│   └── tab10_angle_analysis.py
└── config/
    └── theme.py
```

**Target structure (after Phase 1):**
```
shared_utils/              ← Move utils/ here
├── image_processing.py
├── ...
└── plots.py

streamlit_app/             ← Keep tabs & config
├── app.py
├── tabs/
└── config/

pyqt_app/                  ← Create new
├── main.py
├── backend/
└── gui/
```

---

## Immediate Next Steps for Claude Code

### **Step 0: Restructure Project** (One-time setup)

**Task:** Move `streamlit_app/utils/` to project root as `shared_utils/`

**Commands:**
```bash
# From project root
mkdir shared_utils
cp streamlit_app/utils/* shared_utils/
# Test Streamlit still works with path updates
# If works, remove old utils:
rm -rf streamlit_app/utils
```

**Update imports in all streamlit_app files:**
```python
# Add at top of each file that imports utils
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'shared_utils'))
```

**Test:** `cd streamlit_app && streamlit run app.py`

**Report:** Does Streamlit still work?

### **Step 1: After Restructure - Add Utils Tests**

Then proceed with adding test blocks to each shared_utils/*.py

---

## Summary for Claude Code

**Current situation:**
- Everything in `streamlit_app/`
- Utils are in `streamlit_app/utils/`
- No PyQt6 directory yet

**Goal:**
- `shared_utils/` at root (shared by all)
- `streamlit_app/` (keep working)
- `pyqt_app/` (create new)

**First action:**
Move utils to shared location, update imports, verify Streamlit works.

**Then:**
Add test blocks to shared_utils, extract logic to pyqt_app/backend.

---

## Directory Creation Script

```bash
#!/bin/bash
# Run from project root

echo "Creating PyQt6 directory structure..."

mkdir -p pyqt_app/backend
mkdir -p pyqt_app/gui/dialogs
mkdir -p pyqt_app/resources/icons
mkdir -p pyqt_app/tests
mkdir -p shared_utils
mkdir -p tests/test_data
mkdir -p data/{sessions,exports,calibrations}
mkdir -p docs
mkdir -p scripts

touch pyqt_app/__init__.py
touch pyqt_app/backend/__init__.py
touch pyqt_app/gui/__init__.py
touch shared_utils/__init__.py
touch tests/__init__.py

echo "✅ Directory structure created"
```

Save as `create_structure.sh`, run: `bash create_structure.sh`
