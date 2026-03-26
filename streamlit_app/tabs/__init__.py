"""
Tabs Package
Contains individual tab modules for the Streamlit application
Each tab module has a render() function that displays its content
"""

# Import all tab modules
from . import tab1_setup
from . import tab2_preprocessing
from . import tab3_edge_detection
from . import tab4_analysis
from . import tab5_validation
from . import tab6_esp32_camera

__all__ = [
    'tab1_setup',
    'tab2_preprocessing',
    'tab3_edge_detection',
    'tab4_analysis',
    'tab5_validation',
    'tab6_esp32_camera',
]
