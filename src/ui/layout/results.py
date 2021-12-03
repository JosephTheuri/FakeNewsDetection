from dash import Dash, dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import plotly.express as px
from PIL import Image
import pandas as pd
import numpy as np
from app import app
from dash.exceptions import PreventUpdate
from layout.predict import get_prediction

# Layout
layout = html.Div(children=[
    dcc.Store(id='model-outputs'),

    dcc.Loading([
        html.H2(id='title'), 
        html.P(id='sub_title'),
        html.Div(id='test-div'),

        dbc.Row([
            html.Div([
                dcc.Graph(id='reliability-score')
            ], style={'display': 'inline-block', 'margin-top': 15}),

            html.Div([
                html.Div(children = [
                    html.H3("Why?"),
                    html.Div(id='checklist')
                    ],)
                ], style={'display': 'inline-block', 'margin-left': 50, 'vertical-align': 'top'})
        ], style={'margin-left': 10}),

        dbc.Row([
            html.Div([
                html.Div(children = [
                    html.P("Most dominant words in the article are:"),
                    html.Div(
                        dcc.Graph(id='encoded_image'))
                    ])
                ], style={'margin-left': 10})
        ]),

        dbc.Row([
            html.Div([
                html.Div(children = [
                    html.H3("Further Explanation"),
                    html.Div(id='model_table', style={'width':750})
                    ],)
                ])
        ], style={'margin-left': 10}),

        dbc.Row([
            dbc.Button(
                        'Return  Home', 
                        id='return_home', 
                        n_clicks=0,
                        href="/",
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
                    )
        ])
    ])
], style={'position': 'absolute', 'left': '25%'})

@app.callback(Output('model-outputs', 'data'),
              Input('user-text-input', 'data'))
def check_data(data):
    if data is None:
        raise PreventUpdate
    else:
        # return dummy_inputs(data)
        return get_prediction(data)


@app.callback([Output('title', 'children'), Output('sub_title', 'children')],
              Input('model-outputs', 'data'))
def update_titles(data):
    label = data['label']
    tag = 'reliable' if label == 0 else 'unreliable'
    title = "You are reading {} news!".format(tag)
    sub_title = "We carefully evaluated your news article and found strong indicators that the news is {}. Here are our results:".format(tag)
    return title, sub_title


@app.callback(Output('reliability-score', 'figure'),
              Input('model-outputs', 'data'))
def update_pie(data):
    label, score = data['label'], data['score']

    # Chart title
    tag = 'reliable' if label == 0 else 'unreliable'
    circle_title = '{} News Score'.format(tag.capitalize())

    # Plot chart
    pie = px.pie(
        names=['names'],
        values=[1],
        title=circle_title,
        width=180,
        height=180,
        hole=0.95
        )
    pie.update_layout(
        showlegend=False,
        margin= dict(t = 40,b = 20,l = 10,r = 20),
        title= dict(font = dict(size = 14)),
        hovermode= False,
        annotations= [
            dict(
                text=score,
                x=0.5,
                y=0.5,
                font_size=25,
                showarrow=False
                )
            ]
        )
    pie.update_traces(
        marker=dict(colors=['gray']),
        textinfo='none'
        )

    return pie


@app.callback(Output('checklist', 'children'),
              Input('model-outputs', 'data'))
def update_checklist(data):
    feature_importance = data['fi']
    checklist = dcc.Checklist(
        options=[
            {'label': x, 'value':y, 'disabled':True} for x,y in zip(feature_importance, feature_importance)
        ],
        value=feature_importance,
        labelStyle={'display': 'block', 'margin-bottom': 5}
    )
    return checklist


@app.callback(Output('encoded_image', 'figure'),
              Input('model-outputs', 'data'))
def update_wordcloud(self):
    img_path='D:\JT\GTU OMS\Data and Visual Analytics (CSE 6242)\Project\src\processing\word-cloud.jpg'
    img = np.array(Image.open(img_path))

    image_figure = px.imshow(img, color_continuous_scale='gray')
    image_figure.update_layout(coloraxis_showscale=False)
    image_figure.update_xaxes(showticklabels=False)
    image_figure.update_yaxes(showticklabels=False)
    return image_figure


@app.callback(Output('model_table', 'children'),
              Input('model-outputs', 'data'))
def update_model_table(data):
    model_data = data['model_data']
    model_table = dash_table.DataTable(
        id='table',
        columns=[{"name": i, "id": i} for i in ['Model', 'Results']],
        data=model_data,
        style_cell={'textAlign': 'left'},
        style_data={
            'whiteSpace': 'normal',
            'height': 'auto',
            'minWidth': 100
        },
    )
    return model_table


@app.callback(Output('user-text-input', 'clear_data'),
            [Input('return_home', 'n_clicks')])
def clear_click(n_click_clear):
    if n_click_clear is not None and n_click_clear > 0:
        return True
    return False


if __name__ == '__main__':
    app.run_server(debug=True)