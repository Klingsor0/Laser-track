import dash
import dash_bootstrap_components as dbc
from dash import html, dcc

# Import layouts
from layouts import tab1_layout #, tab2_layout, tab3_layout

# Import callbacks (they auto-register)
from callbacks import tab1_callbacks #, tab2_callbacks, tab3_callbacks

# Initialize app with dark theme
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    suppress_callback_exceptions=True
)

# Color theme (your palette)
THEME = {
    'bg': '#060807',
    'panel': '#30394A',
    'text': '#c4e49a',
    'primary': '#6EBA31',
    'secondary': '#509A24',
    'forest': '#447130',
}

# Main layout
app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col([
            html.H1("🖼️ Image Processing Pipeline", 
                   style={'color': THEME['primary'], 'marginTop': '20px'})
        ])
    ]),
    
    html.Hr(style={'borderColor': THEME['forest']}),
    
    # Tabs
    dcc.Tabs(
        id='main-tabs',
        value='tab-1',
        children=[
            dcc.Tab(label='1️⃣ Setup', value='tab-1',
                   style={'backgroundColor': THEME['panel'], 'color': THEME['text']},
                   selected_style={'backgroundColor': THEME['primary'], 'color': THEME['bg']}),
            dcc.Tab(label='2️⃣ Preprocessing', value='tab-2',
                   style={'backgroundColor': THEME['panel'], 'color': THEME['text']},
                   selected_style={'backgroundColor': THEME['primary'], 'color': THEME['bg']}),
            dcc.Tab(label='3️⃣ Edge Detection', value='tab-3',
                   style={'backgroundColor': THEME['panel'], 'color': THEME['text']},
                   selected_style={'backgroundColor': THEME['primary'], 'color': THEME['bg']}),
        ],
        style={'marginBottom': '20px'}
    ),
    
    # Tab content
    html.Div(id='tab-content'),
    
    # Hidden stores for data persistence
    dcc.Store(id='original-image-store'),
    dcc.Store(id='calibration-store'),
    dcc.Store(id='roi-store'),
    
], fluid=True, style={'backgroundColor': THEME['bg'], 'minHeight': '100vh', 'padding': '20px'})


# Callback to render tab content
@app.callback(
    dash.dependencies.Output('tab-content', 'children'),
    dash.dependencies.Input('main-tabs', 'value')
)
def render_tab_content(active_tab):
    if active_tab == 'tab-1':
        return tab1_layout.layout
    elif active_tab == 'tab-2':
        return tab2_layout.layout
    elif active_tab == 'tab-3':
        return tab3_layout.layout
    return html.Div("Select a tab")


if __name__ == '__main__':
    app.run(debug=True, port=8050)
