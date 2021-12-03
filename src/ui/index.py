from dash import Dash, dcc, html, Input, Output, State
from layout import home, results
from app import app


app.layout = html.Div(children=[
    html.H1(children='Fake News Busters'),

    html.Div(children='''
        A student project by Georgia Tech.
    '''),

    dcc.Location(id='url', refresh=False),
    dcc.Store(id='user-text-input'),

    dcc.Loading(html.Div(id='page-content', children=[]))
])

@app.callback(
        Output(component_id="page-content", component_property='children'),
        Input('url', 'pathname'),
        State('user-text-input', 'data')
    )
def display_page(pathname, data):
    if pathname == '/':
        return home.layout
    elif pathname == '/results':
        return results.layout


if __name__ == '__main__':
    app.run_server(debug=True)
