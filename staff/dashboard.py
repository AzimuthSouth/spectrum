import base64
import datetime
import io
import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']


def get_options(names):
    return [{'label': name, 'value': name} for name in names]


def generate_table(dataframe, max_rows=100):
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), max_rows))
        ])
    ], style={'width': '20%'})


# Load data
df = pd.read_csv('sample.txt')
time = df["Time"]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# fig = px.scatter(df, x="Time", y="Name1")


app.layout = html.Div([
    html.H5("Spectrum Analysis", style={'text-align': 'center'}),
    html.Div([
        html.Div([
            html.Div([
                html.Label('1st signal'),
                dcc.Dropdown(
                    id='x_axe',
                    options=get_options(df.columns),
                    value='Time',
                )], style={'width': '20%', 'display': 'inline-block'}),
            html.Div([
                html.Label('2nd signal'),
                dcc.Dropdown(
                    id='y_axe',
                    options=get_options(df.columns),
                    multi=True,
                    value=['Name1']
                )], style={'width': '20%', 'display': 'inline-block'}),

            dcc.Graph(
                id='input-graph',
                style={'width': '50%'}
            )
        ]),
        html.Table(
            id='input-table',
            style={'width': '50%'}
        )
    ], style={'columnCount': 1})
])


@app.callback(
    Output('input-graph', 'figure'),
    Output('input-table', 'children'),
    Input('x_axe', 'value'),
    Input('y_axe', 'value'))
def update_graph(x_axe, y_axe):
    data = []

    for y in y_axe:
        trace = go.Scatter(x=df[x_axe], y=df[y], mode='lines+markers', name=y)
        data.append(trace)

    # fig = px.line(df, x=x_axe, y=y_axe)
    # fig = px.scatter(df, x=x_axe, y=y_axe)

    layout = go.Layout(xaxis={'title': x_axe},
                       yaxis={'title': 'input'},
                       margin={'l': 40, 'b': 40, 't': 50, 'r': 50},
                       hovermode='closest')

    fig = go.Figure(data=data, layout=layout)

    dff = df[[x_axe] + y_axe]
    tb = generate_table(dff, 100)
    return [fig, tb]


if __name__ == '__main__':
    app.run_server(debug=True)
