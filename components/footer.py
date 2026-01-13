from dash import html, dcc

layout = html.Footer(className="site-footer", children=[
    html.Div(className="footer-content", children=[
        html.Div(className="footer-col", children=[
            html.H4("International Water Management Institute"),
            html.P("Science for a water-secure world.", style={"color": "rgba(255,255,255,0.7)"})
        ]),
        html.Div(className="footer-col", children=[
            html.H4("Quick Links"),
            html.Ul(className="footer-links", children=[
                html.Li(dcc.Link("About", href="/")),
                html.Li(dcc.Link("Framework", href="/framework")),
                html.Li(dcc.Link("Explore", href="/explore")),
            ])
        ]),
        html.Div(className="footer-col", children=[
            html.H4("Contact"),
            html.P("127 Sunil Mawatha, Pelawatte, Battaramulla, Sri Lanka", style={"color": "rgba(255,255,255,0.7)"}),
            html.P("iwmi@cgiar.org", style={"color": "rgba(255,255,255,0.7)"})
        ])
    ]),
    html.Div(className="footer-bottom", children=[
        html.P("© 2024 International Water Management Institute (IWMI). All rights reserved.")
    ])
])
