from dash import html, dcc
import dash_bootstrap_components as dbc

layout = html.Nav(
    className="navbar-custom",
    children=[
        html.Div(className="navbar-brand-group", children=[
            html.Img(src='assets/iwmi.png', className="nav-logo logo-white-filter"),
            html.H1("Water Accounting Jordan", style={"color": "white", "margin": 0, "fontSize": "1.5rem", "fontWeight": "600"}),
        ]),
        html.Div(className="nav-links", children=[
            dcc.Link("About", href="/", className="nav-link"),
            dcc.Link("Framework", href="/framework", className="nav-link"),
            dcc.Link("Explore", href="/explore", className="nav-link"),
            html.Img(src='assets/cgiar.png', className="nav-logo logo-white-filter"),
        ])
    ]
)
