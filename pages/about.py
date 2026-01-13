from dash import html, dcc

layout = html.Div([
    # Hero Section
    html.Div(className="hero-section", children=[
        html.Div(className="hero-bg-overlay"),
        html.Div(className="hero-content", children=[
            html.H1("Rapid Water Accounting Dashboard - Jordan", className="hero-title"),
            html.P("Empowering sustainable water management through advanced remote sensing data and hydrological modeling.", className="hero-subtitle"),
        ])
    ]),
    # Features Section
    html.Div(className="content-section", children=[
        html.Div(className="section-container", children=[
            html.Div(className="section-header", children=[
                html.H2("Key Features", className="section-title"),
                html.P("A comprehensive toolkit for water resource assessment in data-scarce regions.", className="section-desc")
            ]),
            html.Div(className="grid-3", children=[
                html.Div(className="feature-card", children=[
                    html.Div("📊", className="feature-icon"),
                    html.H3("Basin Analysis", className="feature-title"),
                    html.P("Interactive maps and metrics for major basins in Jordan. Analyze inflows, outflows, and storage changes.", className="feature-text")
                ]),
                html.Div(className="feature-card", children=[
                    html.Div("🌧️", className="feature-icon"),
                    html.H3("Climate Data", className="feature-title"),
                    html.P("Visualize long-term precipitation and evapotranspiration trends derived from high-resolution satellite data.", className="feature-text")
                ]),
                html.Div(className="feature-card", children=[
                    html.Div("📑", className="feature-icon"),
                    html.H3("WA+ Reporting", className="feature-title"),
                    html.P("Standardized Water Accounting Plus (WA+) sheets and indicators to support evidence-based decision making.", className="feature-text")
                ])
            ])
        ])
    ]),
])
