from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
from app import app
from dash.exceptions import PreventUpdate

layout =  html.Div(
    children=[
        dcc.Loading(
            id="loading-home",
            children=[
                html.Div([
                    html.Div(
                        dcc.Textarea(
                            id='input-text',
                            minLength = 20,
                            # debounce = True,
                            # persistence_type  = 'memory',
                            placeholder="Paste text here with a minimum of 20 words ...", 
                            style={
                                'height': "40vh",
                                'width': '40vw',
                                'margin': '10px',
                                'text-align': "center",
                                'justifyContent':'center',
                                }
                            )
                        ),

                    dbc.Button(
                        'Reliable or Unreliable? Check now!', 
                        id='submit-text', 
                        n_clicks=0,
                        href="/results",
                        style={
                                'background-color': '#4CAF50',
                                'border-radius': '8px',
                                'color': "white",
                                'padding': "10px 24px",
                                'text-align': "center",
                                'text-decoration': "none",
                                'display': "inline-block",
                                'font-size': '16px',
                                'margin-left': '10vw',
                                'margin': '10px'
                                }
                    ),

                    html.Div(
                        id='container-input-text',
                        children='Enter a value and press submit'
                    )
                    ], style={'position': 'absolute', 'top': '25%', 'left': '25%'})
            ],
            type="circle", style={'position': 'absolute', 'margin-top': '25%', 'left': '45%'}
        )
    ],
    style= {'position': 'relative', 'margin-top':'40px'}
    )

@app.callback(
    Output('user-text-input', 'data'),
    Input('submit-text', 'n_clicks'),
    State('input-text', 'value'))
def show_results(self, data):
    if data is None:
        raise PreventUpdate
    return data


@app.callback(Output('submit-text', 'disabled'),
             [Input('input-text', 'n_blur')], 
             State('input-text', 'value'))
def set_button_enabled_state(n_blur, value):
    if value is None or len(value.split(' ')) < 20:
        return True
    else:
        return False

# if __name__ == '__main__':
#     app.run_server(debug=True)