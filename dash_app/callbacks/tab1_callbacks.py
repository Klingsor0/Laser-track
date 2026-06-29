from dash import Input, Output, State, callback, html
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import base64
import io
from PIL import Image
import numpy as np

THEME = {
    'bg': '#060807',
    'panel': '#30394A',
    'text': '#c4e49a',
    'primary': '#6EBA31',
    'secondary': '#509A24',
    'forest': '#447130',
}

# Callback 1: Handle image upload - UPDATE existing canvas
@callback(
    [Output('calibration-canvas', 'figure'),
     Output('calibration-canvas', 'style'),
     Output('canvas-placeholder', 'style'),
     Output('original-image-store', 'data'),
     Output('upload-status', 'children')],
    Input('upload-image', 'contents'),
    State('upload-image', 'filename')
)
def handle_upload(contents, filename):
    if contents is None:
        # No image uploaded
        return (
            go.Figure(),  # Empty figure
            {'display': 'none'},  # Hide canvas
            {
                'display': 'block',
                'color': THEME['text'],
                'padding': '50px',
                'textAlign': 'center',
                'border': f'2px dashed {THEME["forest"]}',
                'borderRadius': '8px',
                'backgroundColor': THEME['panel']
            },  # Show placeholder
            None,
            ""
        )
    
    # Decode image
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    img = Image.open(io.BytesIO(decoded))
    img_array = np.array(img)
    
    # Store as base64
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode()
    
    # Create figure with image
    fig = go.Figure()
    fig.add_layout_image(
        dict(
            source=f"data:image/png;base64,{img_b64}",
            xref="x", yref="y",
            x=0, y=img_array.shape[0],
            sizex=img_array.shape[1], sizey=img_array.shape[0],
            sizing="stretch",
            layer="below"
        )
    )
    
    fig.update_xaxes(range=[0, img_array.shape[1]], showgrid=False, zeroline=False)
    fig.update_yaxes(range=[0, img_array.shape[0]], showgrid=False, zeroline=False, scaleanchor="x")
    
    fig.update_layout(
        width=img_array.shape[1],
        height=img_array.shape[0],
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor=THEME['bg'],
        plot_bgcolor=THEME['panel'],
        dragmode='drawline',
        newshape=dict(line_color=THEME['primary'], line_width=3)
    )
    
    # Status
    status = dbc.Alert(
        f"✅ Image loaded: {img_array.shape[1]}×{img_array.shape[0]} pixels",
        color="success",
        style={'backgroundColor': THEME['forest'], 'border': 'none', 'color': THEME['text']}
    )
    
    # Show canvas, hide placeholder
    canvas_style = {
        'border': f'2px solid {THEME["forest"]}',
        'borderRadius': '8px',
        'display': 'block'
    }
    placeholder_style = {'display': 'none'}
    
    return fig, canvas_style, placeholder_style, img_b64, status


# Callback 2: Calculate calibration (now calibration-canvas exists!)
@callback(
    Output('calibration-metrics', 'children'),
    [Input('calibration-canvas', 'relayoutData'),
     Input('reference-value', 'value'),
     Input('unit-dropdown', 'value')]
)
def update_calibration_metrics(relayout_data, ref_value, unit):
    if relayout_data is None or 'shapes' not in relayout_data:
        return html.P("👆 Draw a line", style={'color': THEME['text']})
    
    shapes = relayout_data.get('shapes', [])
    if len(shapes) == 0:
        return html.P("👆 Draw a line", style={'color': THEME['text']})
    
    # Get last drawn line
    line = shapes[-1]
    x0, y0 = line['x0'], line['y0']
    x1, y1 = line['x1'], line['y1']
    
    pixel_distance = np.sqrt((x1 - x0)**2 + (y1 - y0)**2)
    
    if pixel_distance == 0:
        return html.P("Line too short", style={'color': 'orange'})
    
    scale = ref_value / pixel_distance
    
    return html.Div([
        html.P(f"Line length: {pixel_distance:.1f} px",
              style={'color': THEME['text'], 'margin': '5px 0'}),
        html.P(f"Scale: {scale:.6f} {unit}/px",
              style={'color': THEME['primary'], 'fontWeight': 'bold', 'margin': '5px 0'})
    ])


# Callback 3: Save calibration
@callback(
    [Output('calibration-store', 'data'),
     Output('calibration-status', 'children')],
    Input('set-calibration-btn', 'n_clicks'),
    [State('calibration-canvas', 'relayoutData'),  # Now this works!
     State('reference-value', 'value'),
     State('unit-dropdown', 'value')],
    prevent_initial_call=True
)
def save_calibration(n_clicks, relayout_data, ref_value, unit):
    if relayout_data is None or 'shapes' not in relayout_data:
        return None, ""
    
    shapes = relayout_data.get('shapes', [])
    if len(shapes) == 0:
        return None, ""
    
    line = shapes[-1]
    x0, y0 = line['x0'], line['y0']
    x1, y1 = line['x1'], line['y1']
    
    pixel_distance = np.sqrt((x1 - x0)**2 + (y1 - y0)**2)
    scale = ref_value / pixel_distance
    
    calibration_data = {
        'scale': scale,
        'unit': unit,
        'reference_value': ref_value,
        'pixel_distance': pixel_distance,
        'line_coords': [x0, y0, x1, y1]
    }
    
    status = dbc.Alert([
        html.H5("✅ Calibration Active", style={'color': THEME['text']}),
        html.P(f"Scale: {scale:.6f} {unit}/pixel", style={'margin': 0}),
        html.P(f"Reference: {ref_value:.3f} {unit} = {pixel_distance:.1f} px", style={'margin': 0})
    ], color="success", style={'backgroundColor': THEME['forest'], 'border': 'none', 'color': THEME['text']})
    
    return calibration_data, status
