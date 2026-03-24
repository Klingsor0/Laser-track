"""
Plotting Utilities
Functions for creating themed Plotly visualizations
"""

import plotly.graph_objects as go
from config.theme import THEME


def create_curve_analysis_plot(x, y, x_range, y_spl, y_spl_1, min_pt, critic_pts,
                               mod_parab, radius_mask, R, model_branch):
    """
    Create a themed Plotly figure for curve analysis
    
    Args:
        x: Data x-coordinates
        y: Data y-coordinates
        x_range: Range for plotting smooth curves
        y_spl: Spline function
        y_spl_1: First derivative of spline
        min_pt: Minimum point [index, x_value]
        critic_pts: Critical points array
        mod_parab: Parabolic model object
        radius_mask: Boolean mask function for parabolic region
        R: Characteristic radius
        model_branch: Branch model function
        
    Returns:
        plotly.graph_objects.Figure: Configured figure object
    """
    fig = go.Figure()

    # Data points
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode='markers',
        marker=dict(
            size=8, 
            color=THEME['secondary'], 
            opacity=0.7,
            line=dict(color=THEME['primary'], width=0.5)
        ),
        name='Data points'
    ))

    # Spline fit
    fig.add_trace(go.Scatter(
        x=x_range, 
        y=y_spl(x_range),
        mode='lines',
        line=dict(width=3, color=THEME['primary']),
        name='Spline fit'
    ))

    # First derivative (scaled for visibility)
    fig.add_trace(go.Scatter(
        x=x_range, 
        y=100*y_spl_1(x_range) + 40,  # Scaled and offset
        mode='lines',
        line=dict(width=1.5, color=THEME['text_dim'], dash='dash'),
        opacity=0.7,
        name='Spline derivative (×100, offset)'
    ))

    # Minimum point
    fig.add_trace(go.Scatter(
        x=[min_pt[1]], 
        y=[y_spl(min_pt[1])],
        mode='markers',
        marker=dict(
            size=14,
            color=THEME['accent'],
            symbol='triangle-down',
            line=dict(color=THEME['text'], width=2)
        ),
        name='Minimum'
    ))

    # Critical points (inflection points)
    fig.add_trace(go.Scatter(
        x=critic_pts, 
        y=y_spl(critic_pts),
        mode='markers',
        marker=dict(
            size=12,
            color=THEME['forest'],
            symbol='diamond',
            line=dict(color=THEME['text'], width=1.5)
        ),
        name='Critical points'
    ))

    # Parabolic model (central region)
    x_parab = x_range[radius_mask(x_range)]
    fig.add_trace(go.Scatter(
        x=x_parab, 
        y=mod_parab.predict(x_parab),
        mode='lines',
        line=dict(width=2.5, color=THEME['dark_green'], dash='dashdot'),
        name='Parabolic model'
    ))

    # Branch model (outer regions)
    x_branch = x_range[~radius_mask(x_range)]
    fig.add_trace(go.Scatter(
        x=x_branch,
        y=model_branch(mod_parab._coeficientes[2], R, x_branch - min_pt[1]),
        mode='lines',
        line=dict(width=2.5, color=THEME['dark_green'], dash='dashdot'),
        name='Branch model',
        showlegend=False  # Same as parabolic, don't duplicate in legend
    ))

    # Apply themed layout
    fig.update_layout(
        title={
            'text': 'Curve Fitting and Critical Point Analysis',
            'font': {'size': 18, 'family': 'monospace', 'color': THEME['primary']},
            'x': 0.5,
            'xanchor': 'center'
        },
        paper_bgcolor=THEME['bg'],
        plot_bgcolor=THEME['panel'],
        font=dict(color=THEME['text'], family='monospace'),
        xaxis=dict(
            title=dict(text='x coordinate (pixels)',
                      font=dict(size=13, color=THEME['primary'])),
            range=[x_range.min() - 5, x_range.max() + 5],
            showgrid=True,
            gridwidth=1,
            zeroline=False,
            tickfont=dict(color=THEME['text'])
        ),
        yaxis=dict(
            title=dict(text='y coordinate (pixels)',
                      font=dict(size=13, color=THEME['primary'])),
            range=[y.min() - 10, y.max() + 10],
            showgrid=True,
            gridwidth=1,
            zeroline=False,
            tickfont=dict(color=THEME['text'])
        ),
        legend=dict(
            x=1.02,
            y=1,
            bgcolor='rgba(48, 57, 74, 0.95)',
            bordercolor=THEME['forest'],
            borderwidth=1,
            font=dict(color=THEME['text'], size=11)
        ),
        hovermode='closest',
        width=900,
        height=600,
        margin=dict(l=60, r=150, t=80, b=60)
    )

    return fig


def create_angle_analysis_plot(x_data, y_data, x_range, model_func, title="Angle Analysis"):
    """
    Create a themed plot for angle vs position analysis
    
    Args:
        x_data: Measured x positions
        y_data: Measured angles
        x_range: Range for model curve
        model_func: Model function for prediction
        title: Plot title
        
    Returns:
        plotly.graph_objects.Figure: Configured figure object
    """
    fig = go.Figure()

    # Data points
    fig.add_trace(go.Scatter(
        x=x_data, 
        y=y_data,
        mode='markers',
        marker=dict(
            size=8, 
            color=THEME['secondary'], 
            opacity=0.7,
            line=dict(color=THEME['primary'], width=0.5)
        ),
        name='Measured data'
    ))

    # Model curve
    fig.add_trace(go.Scatter(
        x=x_range, 
        y=model_func(x_range),
        mode='lines',
        line=dict(width=2.5, color=THEME['dark_green'], dash='dashdot'),
        name='Theoretical model'
    ))

    # Apply themed layout
    fig.update_layout(
        title={
            'text': title,
            'font': {'size': 18, 'family': 'monospace', 'color': THEME['primary']},
            'x': 0.5,
            'xanchor': 'center'
        },
        paper_bgcolor=THEME['bg'],
        plot_bgcolor=THEME['panel'],
        font=dict(color=THEME['text'], family='monospace'),
        xaxis=dict(
            title=dict(text='Mirror position',
                      font=dict(size=13, color=THEME['primary'])),
            showgrid=True,
            gridwidth=1,
            tickfont=dict(color=THEME['text'])
        ),
        yaxis=dict(
            title=dict(text='Angle',
                      font=dict(size=13, color=THEME['primary'])),
            showgrid=True,
            gridwidth=1,
            tickfont=dict(color=THEME['text'])
        ),
        legend=dict(
            bgcolor='rgba(48, 57, 74, 0.95)',
            bordercolor=THEME['forest'],
            borderwidth=1,
            font=dict(color=THEME['text'], size=11)
        ),
        hovermode='closest'
    )

    return fig
