import numpy
import base64
import io
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
from staff import prepare
from staff import analyse
from scipy import signal

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server

app.layout = html.Div([
    html.H4("Spectrum Analysis", style={'text-align': 'center'}),
    html.Hr(),
    dcc.Upload(
        id='upload_data',
        children=html.Div([
            html.A('Select File')
        ]),
    ),
    html.H5(id='filename'),
    html.Label(id='time_range'),

    dcc.Tabs([
        dcc.Tab(label='Expect signal', children=[
            html.Div([
                dcc.Dropdown(
                    id='signal_1'
                )
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
                marks={str(i): str(i) for i in range(1, 11)},
                step=None), style={'width': '50%', 'padding': '0px 20px 20px 20px'}),

            html.Div([
                dcc.Graph(id='input_graph', style={'width': '100%', 'height': '100%'}),
                html.Div([
                    dcc.Input(
                        id='t_start',
                        type='number'
                    ),
                    dcc.Input(
                        id='t_end',
                        type='number'
                    )
                ], style={'width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center'}),

                html.Div([
                    dcc.RangeSlider(
                        id='time_range_slider',
                        allowCross=False
                    ),
                    html.Div(id='output-container-range-slider')
                ]),
                html.Div([
                    html.Label("Resize graph"),
                    dcc.Slider(
                        id='graph_width',
                        min=1,
                        max=15,
                        value=10,
                        marks={str(i): str(i) for i in range(1, 16)},
                        step=None),
                    dcc.Slider(
                        id='graph_height',
                        min=1,
                        max=15,
                        value=8,
                        marks={str(i): str(i) for i in range(1, 16)},
                        step=None)
                ], style={'width': '40%', 'display': 'inline-block'})

            ])

        ]),
        dcc.Tab(label='Expect spectrum', children=[
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

            html.Div(dcc.Graph(id='spectrum_graph'),
                     style={'width': '90vh', 'height': '90vh'}
                     ),
        ]),
        dcc.Tab(label='Expect coherence', children=[
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
            dcc.Input(
                id='segment_len',
                type='number',
                value=256
            ),
            html.Label(id='inspection'),
            dcc.Graph(id='coherence_graph', style={'width': '90vh', 'height': '90vh'}),
        ])
    ]),

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
              Output('t_start', 'min'),
              Output('t_start', 'max'),
              Output('t_start', 'step'),
              Output('t_end', 'min'),
              Output('t_end', 'max'),
              Output('t_end', 'step'),
              Output('time_range_slider', 'min'),
              Output('time_range_slider', 'max'),
              Output('time_range_slider', 'step'),
              Input('upload_data', 'contents'),
              State('upload_data', 'filename'))
def upload_file(contents, filename):
    df = pd.DataFrame()
    time_range = ""
    trp = [0.0, 1.0, 0.5]
    if contents:
        df = parse_data(contents, filename)
        trp = calc_time_range(df)
        time_range = f"Time from {trp[0]} to {trp[1]}, mean time step is {trp[2]}, time step deviation is {trp[3]}"
    options = get_options(df.columns)
    return [df.to_json(date_format='iso', orient='split'),
            options, filename, time_range,
            options[1:], options[1:],
            options[1:], options[1:],
            trp[0], trp[1], trp[2],
            trp[0], trp[1], trp[2],
            trp[0], trp[1], trp[2]]


@app.callback(Output('input_graph', 'figure'),
              Output('time_range_slider', 'value'),
              Output('t_start', 'value'),
              Output('t_end', 'value'),
              Input('signal_1', 'value'),
              Input('signal_filter', 'value'),
              Input('smoothing_window', 'value'),
              Input('graph_width', 'value'),
              Input('graph_height', 'value'),
              Input('t_start', 'value'),
              Input('t_end', 'value'),
              Input('time_range_slider', 'value'),
              State('loading_data', 'children'),
              State('t_start', 'min'),
              State('t_start', 'max'))
def update_graph(signal_1, signal_filter, k, graph_width, graph_height,
                 t_start, t_end, t_range, loading_data, t_min, t_max):
    # set time range if None
    val1 = t_min if t_start is None else t_start
    val2 = t_max if t_end is None else t_end

    # update time range if it changes
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    if trigger_id == "t_start" or trigger_id == 't_end':
        val1 = t_start
        val2 = t_end
    if trigger_id == "time_range_slider":
        val1 = t_range[0]
        val2 = t_range[1]

    data = []
    if signal_1:
        df = pd.read_json(loading_data, orient='split')
        tm = df["Time"]
        sg = df[signal_1]
        sig = []
        tim = []
        for i in range(len(tm)):
            if val1 <= tm[i] <= val2:
                tim.append(tm[i])
                sig.append(sg[i])
        data.append(go.Scatter(x=tim, y=sig, mode='lines+markers', name=signal_1))

        if 'SM' in signal_filter:
            sig = prepare.smoothing(sig, k)
            data.append(go.Scatter(x=tim, y=sig, mode='lines+markers', name='smooth'))

        if 'HW' in signal_filter:
            sig = prepare.correction_hann(sig)
            data.append(go.Scatter(x=tim, y=sig, mode='lines+markers', name='hann_correction'))

    layout = go.Layout(xaxis={'title': 'Time'},
                       yaxis={'title': 'Input'},
                       margin={'l': 40, 'b': 40, 't': 50, 'r': 50},
                       hovermode='closest',
                       width=150 * graph_width, height=100 * graph_height)

    fig = go.Figure(data=data, layout=layout)

    return [fig, [val1, val2], val1, val2]


@app.callback(Output('spectrum_graph', 'figure'),
              # Output('spectrum_graph1', 'figure'),
              Input('spectrum_1', 'value'),
              Input('spectrum_2', 'value'),
              Input('spectrum_filter', 'value'),
              Input('smoothing_window_spectrum', 'value'),
              Input('t_start', 'value'),
              Input('t_end', 'value'),
              State('loading_data', 'children'))
def update_graph(spectrum_1, spectrum_2, spectrum_filter, k, t_start, t_end, loading_data):
    fig = make_subplots(rows=1, cols=2)
    if spectrum_1 and spectrum_2:
        df = pd.read_json(loading_data, orient='split')
        tm = df["Time"]
        val1 = tm[0] if t_start is None else t_start
        val2 = tm[-1] if t_end is None else t_end
        sg1 = df[spectrum_1]
        sg2 = df[spectrum_2]
        sig1 = []
        sig2 = []
        tim = []
        for i in range(len(tm)):
            if val1 <= tm[i] <= val2:
                tim.append(tm[i])
                sig1.append(sg1[i])
                sig2.append(sg2[i])

        if 'SM' in spectrum_filter:
            sig1 = prepare.smoothing(df[spectrum_1], k)
            sig2 = prepare.smoothing(df[spectrum_2], k)

        if 'HW' in spectrum_filter:
            sig1 = prepare.correction_hann(sig1)
            sig2 = prepare.correction_hann(sig2)

        trp = calc_time_range(df)
        f, g_xy = signal.csd(sig1, sig2, (1.0 / trp[2]), window="boxcar", nperseg=len(sig1))
        mod, phase = analyse.cross_spectrum_mod_fas(g_xy)

        fig.add_trace(go.Scatter(x=f, y=mod, mode='lines+markers', name='cross_spectrum'), row=1, col=1)
        fig.add_trace(go.Scatter(x=f, y=phase, mode='lines+markers', name='phase'), row=1, col=2)
    fig.update_xaxes(title_text="Frequencies", row=1, col=1)
    fig.update_xaxes(title_text="Frequencies", row=1, col=2)
    fig.update_yaxes(title_text="Cross Spectrum Module", row=1, col=1)
    fig.update_yaxes(title_text="Cross Spectrum Phase", row=1, col=2)
    fig.update_layout(height=600, width=1400)
    return fig


@app.callback(Output('coherence_graph', 'figure'),
              Output('inspection', 'children'),
              Input('coherence_1', 'value'),
              Input('coherence_2', 'value'),
              Input('coherence_filter', 'value'),
              Input('smoothing_window_coherence', 'value'),
              Input('segment_len', 'value'),
              Input('t_start', 'value'),
              Input('t_end', 'value'),
              State('loading_data', 'children'))
def update_graph(coherence_1, coherence_2, coherence_filter, k, segment_len, t_start, t_end, loading_data):
    data = []
    f = [-1.0]
    if coherence_1 and coherence_2:
        df = pd.read_json(loading_data, orient='split')
        tm = df["Time"]
        val1 = tm[0] if t_start is None else t_start
        val2 = tm[-1] if t_end is None else t_end
        sg1 = df[coherence_1]
        sg2 = df[coherence_2]
        sig1 = []
        sig2 = []
        tim = []
        for i in range(len(tm)):
            if val1 <= tm[i] <= val2:
                tim.append(tm[i])
                sig1.append(sg1[i])
                sig2.append(sg2[i])

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
