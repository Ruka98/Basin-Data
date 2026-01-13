from dash import dcc, html
import dash_bootstrap_components as dbc
import os

BASIN_DIR = os.path.join(os.getcwd(), "basins")
basin_folders = [d for d in os.listdir(BASIN_DIR) if os.path.isdir(os.path.join(BASIN_DIR, d))] if os.path.isdir(BASIN_DIR) else []
basin_options = [{"label": "Select a Basin...", "value": "none"}] + [{"label": b, "value": b} for b in sorted(basin_folders)]

layout = html.Div(className="dashboard-container", children=[
    html.Div(
        className="filters-panel",
        style={"marginTop": "20px"},
        children=[
            html.Div(style={"maxWidth": "1200px", "margin": "0 auto"}, children=[
                html.H3("Select Basin", style={"color": "#315F83", "marginBottom": "20px", "fontWeight": "600", "fontSize": "1.8rem"}),
                html.Div([
                    html.Div([
                         html.Label("Choose from list:", style={"fontWeight": "bold", "marginBottom": "10px", "display": "block", "color": "#315F83"}),
                         dcc.Dropdown(
                            id="basin-dropdown",
                            options=basin_options,
                            value=None,
                            placeholder="Select a basin...",
                            style={"borderRadius": "4px"}
                        ),
                        html.Div(id="study-area-container", style={"marginTop": "20px", "padding": "20px", "backgroundColor": "#f0f4f8", "borderRadius": "8px", "fontSize": "1rem", "lineHeight": "1.8", "color": "#2c3e50", "textAlign": "justify", "borderLeft": "4px solid #315F83"})
                    ], style={"width": "30%", "display": "inline-block", "verticalAlign": "top"}),

                    html.Div([
                         dcc.Graph(id="basin-map", style={"height": "400px", "borderRadius": "8px", "overflow": "hidden"})
                    ], style={"width": "68%", "display": "inline-block", "marginLeft": "2%", "verticalAlign": "top", "boxShadow": "0 4px 12px rgba(0,0,0,0.1)", "borderRadius": "8px"})
                ])
            ])
        ]
    ),
    html.Div(id="dynamic-content")
])
