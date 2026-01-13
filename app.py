import dash
import dash_bootstrap_components as dbc
from dash import dcc, html
from dash.dependencies import Input, Output

from pages import about, framework, explore
from components import header, footer

app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    header.layout,
    html.Div(id='page-content'),
    footer.layout
])

@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/framework':
        return framework.layout
    elif pathname == '/explore':
        return explore.layout
    else:
        return about.layout

if __name__ == '__main__':
    app.run_server(debug=True)
