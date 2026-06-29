import dash_bootstrap_components as dbc
from dash import html, dcc
import plotly.graph_objects as go

THEME = {
    'bg': '#060807',
    'panel': '#30394A',
    'text': '#c4e49a',
    'primary': '#6EBA31',
    'secondary': '#509A24',
    'forest': '#447130',
}

layout = dbc.Container([
    # Upload Section
    dbc.Row([
        dbc.Col([
            html.H3("1️⃣ Upload Image", style={'color': THEME['primary']}),
            dcc.Upload(
                id='upload-image',
                children=html.Div([
                    '📁 Drag and Drop or ',
                    html.A('Select Files', style={'color': THEME['primary'], 'textDecoration': 'underline'})
                ]),
                style={
                    'width': '100%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '2px',
                    'borderStyle': 'dashed',
                    'borderRadius': '8px',
                    'borderColor': THEME['forest'],
                    'textAlign': 'center',
                    'backgroundColor': THEME['panel'],
                    'color': THEME['text']
                },
                multiple=False
            ),
            html.Div(id='upload-status', style={'marginTop': '10px'})
        ], width=12)
    ], style={'marginBottom': '30px'}),
    
    html.Hr(style={'borderColor': THEME['forest']}),
    
    # Calibration Section
    dbc.Row([
        dbc.Col([
            html.H3("2️⃣ Spatial Calibration", style={'color': THEME['primary']}),
            
            dbc.Accordion([
                dbc.AccordionItem([
                    html.P("1. Draw a line along a known distance", style={'color': THEME['text']}),
                    html.P("2. Enter the real-world measurement", style={'color': THEME['text']}),
                    html.P("3. Click 'Set Calibration'", style={'color': THEME['text']}),
                    html.P("💡 Choose a clear reference (ruler, scale bar)", 
                          style={'color': THEME['secondary'], 'fontStyle': 'italic'}),
                ], title="📏 How to calibrate", style={'backgroundColor': THEME['panel'], 'color': THEME['text']})
            ], start_collapsed=True, style={'marginBottom': '20px'}),
            
        ], width=12)
    ]),
    
    # Canvas + Controls
    dbc.Row([
        # Canvas column
        dbc.Col([
            html.Div([
                html.P("Draw reference line:", style={'color': THEME['text'], 'fontWeight': 'bold'}),
                
                # Canvas created in layout (not callback!)
                dcc.Graph(
                    id='calibration-canvas',
                    figure=go.Figure(),  # Empty initially
                    config={
                        'modeBarButtonsToAdd': ['drawline'],
                        'displaylogo': False
                    },
                    style={
                        'border': f'2px solid {THEME["forest"]}', 
                        'borderRadius': '8px',
                        'display': 'none',  # Hidden initially
                        'minHeight': '400px'
                    }
                ),
                
                html.Div(
                    id='canvas-placeholder',
                    children="👆 Upload an image first",
                    style={
                        'color': THEME['text'], 
                        'padding': '50px',
                        'textAlign': 'center',
                        'border': f'2px dashed {THEME["forest"]}',
                        'borderRadius': '8px',
                        'backgroundColor': THEME['panel']
                    }
                )
            ])
        ], width=8),
        
        # Controls column
        dbc.Col([
            html.Div([
                html.P("Parameters:", style={'color': THEME['text'], 'fontWeight': 'bold'}),
                
                html.Label("Known distance:", style={'color': THEME['text']}),
                dbc.Input(
                    id='reference-value',
                    type='number',
                    value=10.0,
                    min=0.001,
                    step=0.1,
                    style={'backgroundColor': THEME['panel'], 'color': THEME['text'], 'border': f'1px solid {THEME["forest"]}'}
                ),
                
                html.Label("Unit:", style={'color': THEME['text'], 'marginTop': '10px'}),
                dcc.Dropdown(
                    id='unit-dropdown',
                    options=[
                        {'label': 'mm', 'value': 'mm'},
                        {'label': 'cm', 'value': 'cm'},
                        {'label': 'm', 'value': 'm'},
                        {'label': 'µm', 'value': 'µm'},
                        {'label': 'inches', 'value': 'inches'},
                    ],
                    value='mm',
                    style={'backgroundColor': THEME['panel'], 'color': THEME['bg']}
                ),
                
                html.Hr(style={'borderColor': THEME['forest'], 'marginTop': '20px'}),
                
                html.Div(id='calibration-metrics'),
                
                html.Hr(style={'borderColor': THEME['forest']}),
                
                dbc.Button(
                    "🎯 Set Calibration",
                    id='set-calibration-btn',
                    color='success',
                    className='w-100 mb-2',
                    style={'backgroundColor': THEME['primary'], 'border': 'none'}
                ),
                
                dbc.Button(
                    "🔄 Redraw",
                    id='redraw-btn',
                    color='secondary',
                    className='w-100',
                    style={'backgroundColor': THEME['forest'], 'border': 'none'}
                ),
                
            ], style={'padding': '20px', 'backgroundColor': THEME['panel'], 'borderRadius': '8px'})
        ], width=4)
    ], style={'marginBottom': '30px'}),
    
    # Calibration status
    dbc.Row([
        dbc.Col([
            html.Div(id='calibration-status')
        ])
    ]),
    
], fluid=True, style={'color': THEME['text']})
