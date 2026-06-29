# Image Processing Pipeline - Streamlit Application

A modular Streamlit application for image processing, edge detection, and curve analysis with active contour optimization.

## Project Structure

```
.
├── app.py                          # Main application entry point
├── config/
│   ├── __init__.py
│   └── theme.py                    # Color theme and CSS styling
├── tabs/
│   ├── __init__.py
│   ├── tab1_setup.py              # Image upload, calibration, ROI selection
│   ├── tab2_preprocessing.py       # Image filtering and preprocessing
│   ├── tab3_edge_detection.py      # Active contour curve extraction
│   ├── tab4_analysis.py           # Curve fitting and model comparison
│   └── tab5_validation.py         # Experimental validation
├── utils/
│   ├── __init__.py
│   ├── image_processing.py        # Image manipulation functions
│   ├── canvas_utils.py            # Canvas data processing
│   ├── curve_analysis.py          # Mathematical curve fitting
│   └── plotting.py                # Plotly visualization functions
├── utils/snake1.py                 # Active contour (snake) algorithm (your existing file)
├── utils/models.py                 # Model classes (your existing file)
└── utils/plots.py                  # Matplotlib plotting (your existing file)
```

## Features

### Tab 1: Setup
- **Image Upload**: Load images for analysis (JPG, PNG)
- **Spatial Calibration**: Draw a reference line and set real-world scale
- **ROI Selection**: Define region of interest using polygon drawing

### Tab 2: Preprocessing
- **Gaussian Blur**: Smooth images to reduce noise
- **Contrast Adjustment**: Enhance or reduce image contrast
- **Morphological Operations**: Erosion, dilation, opening, closing
- **Histogram Analysis**: Compare before/after filtering

### Tab 3: Edge Detection
- **Manual Initialization**: Draw initial curve approximation
- **Active Contour Optimization**: Greedy energy minimization
- **Parameter Tuning**: Adjust elasticity, rigidity, and edge attraction
- **Export**: Save extracted curve coordinates (CSV, NPY)

### Tab 4: Analysis
- **Spline Fitting**: Smooth curve representation with derivatives
- **Critical Point Detection**: Find minima and inflection points
- **Model Comparison**: Fit parabolic and branch models
- **Dimensionless Scaling**: Transform to universal coordinates
- **Publication Export**: Generate Physical Review-compliant figures

### Tab 5: Validation
- **Experimental Data**: Load angle measurement data
- **Model Fitting**: Piecewise linear regression
- **Theoretical Comparison**: Compare with analytical predictions
- **Residual Analysis**: Assess model accuracy

## Installation

```bash
# Clone repository
git clone <your-repo-url>
cd <your-repo-name>

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Run the application
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

## Workflow

1. **Tab 1**: Upload image, optionally calibrate scale, select ROI
2. **Tab 2**: Apply filters to prepare image for edge detection
3. **Tab 3**: Draw initial curve, optimize with active contour, export coordinates
4. **Tab 4**: Analyze curve, fit models, generate publication figures
5. **Tab 5**: Validate theoretical model against experimental measurements

## Session State Variables

The application uses Streamlit's session state to share data between tabs:

- `original_image`: Uploaded image
- `calibration`: Spatial calibration parameters
- `roi_image`: Cropped region of interest
- `roi_mask`: Binary mask for ROI
- `processed_image`: Filtered image
- `optimized_snake`: Extracted curve coordinates
- `analysis_data`: Curve data for analysis
- `validation_data`: Experimental validation dataset

## Dependencies

- streamlit
- opencv-python (cv2)
- numpy
- pandas
- plotly
- scipy
- scikit-learn
- PIL (Pillow)
- matplotlib
- streamlit-drawable-canvas

## Configuration

### Theme Customization

Edit `config/theme.py` to change the color palette:

```python
THEME = {
    'bg': '#060807',           # Background color
    'primary': '#6EBA31',      # Primary accent color
    'text': '#c4e49a',         # Text color
    # ... more colors
}
```

### Adding New Tabs

1. Create new file in `tabs/` directory (e.g., `tab6_newfeature.py`)
2. Implement `render()` function
3. Import in `tabs/__init__.py`
4. Add to `app.py`:
   ```python
   tab6 = st.tabs(["...", "6️⃣ New Feature"])
   with tab6:
       tab6_newfeature.render()
   ```

## Code Organization

### Utils Modules

- **image_processing.py**: Image loading, cropping, rotation, filtering
- **canvas_utils.py**: Extract polygon points from drawable canvas
- **curve_analysis.py**: Spline fitting, critical points, model comparison
- **plotting.py**: Reusable Plotly visualization functions

### Tab Modules

Each tab module:
- Has a single `render()` function
- Accesses data via `st.session_state`
- Uses utilities from `utils/` and `config/`
- Handles its own UI and logic

## Best Practices

1. **Caching**: Use `@st.cache_data` for expensive operations
2. **Session State**: Store intermediate results for reuse across tabs
3. **Error Handling**: Check if required data exists before processing
4. **User Feedback**: Provide clear messages (success, warnings, errors)
5. **Documentation**: Add docstrings to all functions

## Troubleshooting

### Common Issues

**Import errors:**
```bash
# Make sure you're in the project root directory
cd /path/to/project
streamlit run app.py
```

**Missing dependencies:**
```bash
pip install -r requirements.txt
```

**Canvas not appearing:**
- Check browser console for errors
- Try refreshing the page
- Ensure streamlit-drawable-canvas is installed

## Contributing

1. Create a new branch for your feature
2. Follow existing code style and structure
3. Add docstrings to new functions
4. Test thoroughly before committing

## License

[Your License Here]

## Contact

[Your Contact Information]
