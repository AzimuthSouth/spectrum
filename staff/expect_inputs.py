import numpy
import base64
import io
import plotly.graph_objs as go
import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import prepare
import analyse
from scipy import signal

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    html.H4("Spectrum Analysis", style={'text-align': 'center'}),
    dcc.Upload(
        id='upload_data',
        children=html.Div([
            html.A('Select File'),
        ]),
    ),
    html.H5(id='filename'),
    html.Label(id='time_range'),

    html.Div([
        html.H6('Expect signal'),
        html.Div([
            dcc.Dropdown(
                id='signal_1'
            ),
        ], style={'display': 'inline-block', 'width': '20%'}),

        dcc.Checklist(
            id='signal_filter',
            options=[
                {'label': 'smoothing and centering', 'value': 'SM'},
                {'label': 'hann-weighting', 'value': 'HW'}
            ],
            value=['SM', 'HW'],
            labelStyle={'display': 'inline-block'}
        ),
        html.Label("Smoothing window size"),
        html.Div(dcc.Slider(
            id='smoothing_window',
            min=1,
            max=10,
            value=3,
            marks={str(i): str(i) for i in range(1, 10)},
            step=None), style={'width': '50%', 'padding': '0px 20px 20px 20px'})
    ]),

    dcc.Graph(id='input_graph', style={'width': '50%'}),
    html.Div(id='output-data-upload'),

    html.Div([
        html.H6('Expect spectrum'),
        html.Div([
            dcc.Dropdown(
                id='spectrum_1'
            ),
            dcc.Dropdown(
                id='spectrum_2'
            )
        ], style={'display': 'inline-block', 'width': '20%'}),

        dcc.Checklist(
            id='spectrum_filter',
            options=[
                {'label': 'smoothing and centering', 'value': 'SM'},
                {'label': 'hann-weighting', 'value': 'HW'}
            ],
            value=['SM', 'HW'],
            labelStyle={'display': 'inline-block'}
        ),
        html.Label("Smoothing window size"),
        html.Div(dcc.Slider(
            id='smoothing_window_spectrum',
            min=1,
            max=10,
            value=3,
            marks={str(i): str(i) for i in range(0, 10)},
            step=None), style={'width': '50%', 'padding': '0px 20px 20px 20px'}),
    ]),

    html.Div([dcc.Graph(id='spectrum_graph'),
              dcc.Graph(id='spectrum_graph1'),
              ], style={'display': 'inline-block'}),

    html.Div([
        html.H6('Expect coherence'),
        html.Div([
            dcc.Dropdown(
                id='coherence_1'
            ),
            dcc.Dropdown(
                id='coherence_2'
            )
        ], style={'display': 'inline-block', 'width': '20%'}),

        dcc.Checklist(
            id='coherence_filter',
            options=[
                {'label': 'smoothing and centering', 'value': 'SM'},
                {'label': 'hann-weighting', 'value': 'HW'}
            ],
            value=['SM', 'HW'],
            labelStyle={'display': 'inline-block'}
        ),
        html.Label("Smoothing window size"),
        html.Div(dcc.Slider(
            id='smoothing_window_coherence',
            min=1,
            max=10,
            value=3,
            marks={str(i): str(i) for i in range(0, 10)},
            step=None), style={'width': '50%', 'padding': '0px 20px 20px 20px'}),
        html.Label("Points per segment"),
        html.Div(dcc.Slider(
            id='segment_len',
            min=0,
            max=2048,
            value=256,
            marks={str(128 * i): str(128 * i) for i in range(0, 20)},
            step=None), style={'width': '50%', 'padding': '0px 20px 20px 20px'}),
        html.Label(id='inspection')
    ]),

    dcc.Graph(id='coherence_graph', style={'width': '50%'}),

    html.Div(id='loading_data', style={'display': 'none'})
])


def get_options(names):
    return [{'label': name, 'value': name} for name in names]


def parse_data(contents, filename):
    content_type, content_string = contents.split(',')
    df = pd.DataFrame()
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' or 'txt' in filename:
            # Assume that the user uploaded a CSV or TXT file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    return df


def calc_time_range(df):
    time = df["Time"].to_numpy()
    delta = abs(time[:-1] - time[1:])
    return [min(time), max(time), numpy.mean(delta), numpy.std(delta)]


@app.callback(Output('loading_data', 'children'),
              Output('signal_1', 'options'),
              Output('filename', 'children'),
              Output('time_range', 'children'),
              Output('spectrum_1', 'options'),
              Output('spectrum_2', 'options'),
              Output('coherence_1', 'options'),
              Output('coherence_2', 'options'),
              Output('segment_len', 'max'),
              Output('segment_len', 'marks'),
              Input('upload_data', 'contents'),
              State('upload_data', 'filename'))
def upload_file(contents, filename):
    df = pd.DataFrame()
    time_range = ""
    max_point = 2048
    marks = {str(128 * i): str(128 * i) for i in range(0, 20)}
    if contents:
        df = parse_data(contents, filename)
        trp = calc_time_range(df)
        time_range = f"Time from {trp[0]} to {trp[1]}, mean time step is {trp[2]}, time step deviation is {trp[3]}"
        max_point = df.shape[0]
        if max_point <= 128:
            rng = range(0, max_point + 1)
            marks = {str(i): str(i) for i in rng}
        else:
            rng = range(0, 128 * int(1 + numpy.ceil(max_point / 128)))
            marks = {str(128 * i): str(128 * i) for i in rng}
    options = get_options(df.columns)
    return [df.to_json(date_format='iso', orient='split'),
            options, filename, time_range,
            options[1:], options[1:],
            options[1:], options[1:],
            max_point, marks]


@app.callback(Output('input_graph', 'figure'),
              Input('signal_1', 'value'),
              Input('signal_filter', 'value'),
              Input('smoothing_window', 'value'),
              State('loading_data', 'children'))
def update_graph(signal_1, signal_filter, k, loading_data):
    data = []
    if signal_1:
        df = pd.read_json(loading_data, orient='split')
        sig = df[signal_1]
        data.append(go.Scatter(x=df["Time"], y=sig, mode='lines+markers', name=signal_1))

        if 'SM' in signal_filter:
            sig = prepare.smoothing(sig, k)
            data.append(go.Scatter(x=df["Time"], y=sig, mode='lines+markers', name='smooth'))

        if 'HW' in signal_filter:
            sig = prepare.correction_hann(sig)
            data.append(go.Scatter(x=df["Time"], y=sig, mode='lines+markers', name='hann_correction'))

    layout = go.Layout(xaxis={'title': 'Time'},
                       yaxis={'title': 'Input'},
                       margin={'l': 40, 'b': 40, 't': 50, 'r': 50},
                       hovermode='closest')

    fig = go.Figure(data=data, layout=layout)

    return fig


@app.callback(Output('spectrum_graph', 'figure'),
              Output('spectrum_graph1', 'figure'),
              Input('spectrum_1', 'value'),
              Input('spectrum_2', 'value'),
              Input('spectrum_filter', 'value'),
              Input('smoothing_window_spectrum', 'value'),
              State('loading_data', 'children'))
def update_graph(spectrum_1, spectrum_2, spectrum_filter, k, loading_data):
    data1 = []
    data2 = []
    if spectrum_1 and spectrum_2:
        df = pd.read_json(loading_data, orient='split')
        sig1 = df[spectrum_1]
        sig2 = df[spectrum_2]

        if 'SM' in spectrum_filter:
            sig1 = prepare.smoothing(df[spectrum_1], k)
            sig2 = prepare.smoothing(df[spectrum_2], k)

        if 'HW' in spectrum_filter:
            sig1 = prepare.correction_hann(sig1)
            sig2 = prepare.correction_hann(sig2)

        trp = calc_time_range(df)
        f, g_xy = signal.csd(sig1, sig2, (1.0 / trp[2]), window="boxcar", nperseg=len(sig1))
        mod, phase = analyse.cross_spectrum_mod_fas(g_xy)

        data1.append(go.Scatter(x=f, y=mod, mode='lines+markers', name='cross_spectrum'))
        data2.append(go.Scatter(x=f, y=phase, mode='lines+markers', name='phase'))

    layout1 = go.Layout(xaxis={'title': 'Frequencies'},
                        yaxis={'title': 'Cross_spectrum'},
                        margin={'l': 40, 'b': 40, 't': 50, 'r': 50},
                        hovermode='closest')

    layout2 = go.Layout(xaxis={'title': 'Frequencies'},
                        yaxis={'title': 'Phase'},
                        margin={'l': 40, 'b': 40, 't': 50, 'r': 50},
                        hovermode='closest')

    fig1 = go.Figure(data=data1, layout=layout1)
    fig2 = go.Figure(data=data2, layout=layout2)

    return [fig1, fig2]


@app.callback(Output('coherence_graph', 'figure'),
              Output('inspection', 'children'),
              Input('coherence_1', 'value'),
              Input('coherence_2', 'value'),
              Input('coherence_filter', 'value'),
              Input('smoothing_window_coherence', 'value'),
              Input('segment_len', 'value'),
              State('loading_data', 'children'))
def update_graph(coherence_1, coherence_2, coherence_filter, k, segment_len, loading_data):
    data = []
    f = [-1.0]
    if coherence_1 and coherence_2:
        df = pd.read_json(loading_data, orient='split')
        sig1 = df[coherence_1]
        sig2 = df[coherence_2]

        if 'SM' in coherence_filter:
            sig1 = prepare.smoothing(df[coherence_1], k)
            sig2 = prepare.smoothing(df[coherence_2], k)

        if 'HW' in coherence_filter:
            sig1 = prepare.correction_hann(sig1)
            sig2 = prepare.correction_hann(sig2)

        trp = calc_time_range(df)
        f, c_xx = signal.coherence(sig1, sig2, (1.0 / trp[2]), window="boxcar", nperseg=segment_len)

        data.append(go.Scatter(x=f, y=c_xx, mode='lines+markers', name='cross_spectrum'))

    layout = go.Layout(xaxis={'title': 'Frequencies'},
                       yaxis={'title': 'Coherence'},
                       margin={'l': 40, 'b': 40, 't': 50, 'r': 50},
                       hovermode='closest')

    fig = go.Figure(data=data, layout=layout)

    return [fig, f[-1]]


if __name__ == '__main__':
    app.run_server(debug=True)
