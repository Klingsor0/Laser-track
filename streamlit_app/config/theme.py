"""
Theme Configuration
Contains color palette and CSS styling for the application
"""

import streamlit as st

# ============= COLOR PALETTE =============
# Single source of truth for all colors used in the app
THEME = {
    'bg': '#060807',           # Main background
    'panel': '#30394A',        # Panel/card background
    'text': '#c4e49a',         # Primary text color
    'text_dim': '#899f6b',     # Dimmed text
    'primary': '#6EBA31',      # Lime green - primary accent
    'secondary': '#509A24',    # Bright green
    'accent': '#368912',       # Medium green
    'dark_green': '#335926',   # Dark green
    'forest': '#447130',       # Forest green
    'grid': 'rgba(68, 113, 48, 0.2)',  # Grid lines with transparency
}


def apply_theme():
    """
    Apply custom CSS theme to the Streamlit app
    This function should be called once in the main app.py
    """
    st.markdown("""
    <style>
        /* Main background */
        .stApp {
            background-color: #060807;
            color: #c4e49a;
        }
        
        /* Sidebar */
        [data-testid="stSidebar"] {
            background-color: #30394A;
            border-right: 2px solid #6EBA31;
        }
        
        /* Headers */
        h1 {
            color: #6EBA31 !important;
            text-shadow: 0 0 15px rgba(110, 186, 49, 0.5);
        }
        
        h2 {
            color: #509A24 !important;
        }
        
        h3 {
            color: #c4e49a !important;
        }
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background-color: #30394A;
            border-radius: 8px;
            padding: 4px;
        }
        
        .stTabs [data-baseweb="tab"] {
            background-color: transparent;
            border-radius: 4px;
            color: #c4e49a;
            font-weight: 500;
        }
        
        .stTabs [aria-selected="true"] {
            background: linear-gradient(90deg, #447130 0%, #509A24 100%);
            color: #c4e49a !important;
            box-shadow: 0 0 10px rgba(110, 186, 49, 0.3);
        }
        
        /* Buttons */
        .stButton button {
            background: linear-gradient(90deg, #447130 0%, #6EBA31 100%);
            color: #060807;
            border: none;
            border-radius: 8px;
            font-weight: bold;
            box-shadow: 0 0 15px rgba(110, 186, 49, 0.4);
            transition: all 0.3s;
        }
        
        .stButton button:hover {
            background: linear-gradient(90deg, #509A24 0%, #6EBA31 100%);
            box-shadow: 0 0 25px rgba(110, 186, 49, 0.6);
            transform: translateY(-2px);
        }
        
        /* Primary button */
        .stButton button[kind="primary"] {
            background: linear-gradient(90deg, #368912 0%, #509A24 100%);
            box-shadow: 0 0 20px rgba(80, 154, 36, 0.5);
        }
        
        /* Sliders */
        .stSlider [data-baseweb="slider"] [role="slider"] {
            background-color: #6EBA31;
        }
        
        .stSlider [data-baseweb="slider"] [data-testid="stTickBar"] > div {
            background: linear-gradient(90deg, #335926 0%, #6EBA31 100%);
        }
        
        /* Metrics */
        [data-testid="stMetricValue"] {
            color: #6EBA31;
            font-size: 2rem;
            text-shadow: 0 0 10px rgba(110, 186, 49, 0.5);
        }
        
        [data-testid="stMetricDelta"] {
            color: #509A24;
        }
        
        /* Dataframes */
        .stDataFrame {
            border: 1px solid #447130;
            border-radius: 8px;
        }
        
        /* Text inputs */
        .stTextInput input, .stNumberInput input {
            background-color: #30394A;
            color: #c4e49a;
            border: 1px solid #447130;
        }
        
        .stTextInput input:focus, .stNumberInput input:focus {
            border-color: #6EBA31;
            box-shadow: 0 0 10px rgba(110, 186, 49, 0.3);
        }
        
        /* Select boxes */
        .stSelectbox [data-baseweb="select"] {
            background-color: #30394A;
            border-color: #447130;
        }
        
        /* Info/Success/Warning boxes */
        .stAlert {
            background-color: #30394A;
            border-left: 4px solid #6EBA31;
            color: #c4e49a;
        }
        
        /* Success */
        [data-baseweb="notification"][kind="success"] {
            background-color: #335926;
            border-left: 4px solid #509A24;
        }
        
        /* Warning */
        [data-baseweb="notification"][kind="warning"] {
            background-color: #30394A;
            border-left: 4px solid #368912;
        }
        
        /* Code blocks */
        .stCodeBlock {
            background-color: #30394A !important;
            border: 1px solid #447130;
            border-radius: 8px;
        }
        
        /* Expander */
        .streamlit-expanderHeader {
            background-color: #30394A;
            color: #c4e49a;
            border: 1px solid #447130;
        }
        
        /* Progress bar */
        .stProgress > div > div {
            background: linear-gradient(90deg, #335926 0%, #6EBA31 100%);
        }
        
        /* File uploader */
        [data-testid="stFileUploadDropzone"] {
            background-color: #30394A;
            border: 2px dashed #447130;
        }
        
        [data-testid="stFileUploadDropzone"]:hover {
            border-color: #6EBA31;
        }
    </style>
    """, unsafe_allow_html=True)
