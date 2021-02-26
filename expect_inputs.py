import base64
import io
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
from staff import prepare
from staff import analyse
from staff import schematisation
from staff import pipeline
from scipy import signal
import urllib

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server


def mark(i):
    if i % 10 == 0 or i == 1:
        return "{}%".format(i)
    return ''


app.layout = html.Div([
    html.H4("Spectrum Analysis", style={'text-align': 'center'}),
    html.Hr(),

    html.Div([
        dcc.Upload(
            id='upload_data',
            children=html.Div([
                html.A('Select File')
            ]),
        ),
        html.H5(id='filename'),
        html.Label("Time range parameters", id='time_range'),
    ]),

    # html.Label(id='time_range'),

    dcc.Tabs([
        dcc.Tab(label='Expect signal', children=[
            html.Div([
                dcc.Dropdown(
                    id='signal_1',
                    multi=True
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

            dcc.RadioItems(
                id='graph_lines',
                options=[
                    {'label': 'lines', 'value': 'LL'},
                    {'label': 'lines+markers', 'value': 'LM'}
                ],
                value='LM',
                labelStyle={'display': 'inline-block'}
            ),

            html.Label("Smoothing window size"),
            html.Div([
                dcc.Slider(
                    id='smoothing_window',
                    min=1,
                    max=300,
                    value=3,
                    step=1),
                dcc.Input(
                    id='smoothing_window_input',
                    type='number',
                    min=1,
                    max=300,
                    step=1,
                    value=3
                )
            ], style={'columns': 2}),

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
                    ),

                ], style={'width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center'}),

                html.Div([
                    dcc.RangeSlider(
                        id='time_range_slider',
                        allowCross=False
                    )
                ]),
                html.Button('Export signals', id='pick_signals'),
                html.A('Export signals',
                       id='link-signals',
                       download="data.txt",
                       href="",
                       target="_blank",
                       hidden=True,
                       style={'textAlign': 'right'}),
                dcc.Checklist(
                    id='all_signal',
                    options=[{'label': 'all signals', 'value': 'ALL'}]
                    ),

                html.Hr(),
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
            dcc.RadioItems(
                id='spectrum_lines',
                options=[
                    {'label': 'lines', 'value': 'LL'},
                    {'label': 'lines+markers', 'value': 'LM'}
                ],
                value='LM',
                labelStyle={'display': 'inline-block'}
            ),
            html.Label("Smoothing window size"),

            html.Div([
                dcc.Slider(
                    id='smoothing_window_spectrum',
                    min=1,
                    max=300,
                    value=3,
                    step=1),
                dcc.Input(
                    id='smoothing_window_input_spectrum',
                    type='number',
                    min=1,
                    max=300,
                    step=1,
                    value=3
                )
            ], style={'columns': 2}),

            html.Div(dcc.Graph(id='spectrum_graph'),
                     style={'width': '100%', 'height': '100%'}
                     ),
            html.Hr(),
            html.Button('Export spectrum', id='pick_spectrum'),
            html.A('Export cross spectrum',
                   id='link-spectrum',
                   download="cross_spectrum.txt",
                   href="",
                   target="_blank",
                   hidden=True,
                   style={'textAlign': 'right'}),
            dcc.Checklist(
                    id='all_spectrum',
                    options=[{'label': 'all signals', 'value': 'ALL'}]
                    ),

            html.Div([
                html.Label("Resize graph"),
                dcc.Slider(
                    id='graph_width1',
                    min=1,
                    max=15,
                    value=10,
                    marks={str(i): str(i) for i in range(1, 16)},
                    step=None),
                dcc.Slider(
                    id='graph_height1',
                    min=1,
                    max=15,
                    value=8,
                    marks={str(i): str(i) for i in range(1, 16)},
                    step=None)
            ], style={'width': '40%'})

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
            dcc.RadioItems(
                id='coherence_lines',
                options=[
                    {'label': 'lines', 'value': 'LL'},
                    {'label': 'lines+markers', 'value': 'LM'}
                ],
                value='LM',
                labelStyle={'display': 'inline-block'}
            ),
            html.Label("Smoothing window size"),
            html.Div([
                dcc.Slider(
                    id='smoothing_window_coherence',
                    min=1,
                    max=300,
                    value=3,
                    step=1),
                dcc.Input(
                    id='smoothing_window_input_coherence',
                    type='number',
                    min=1,
                    max=300,
                    step=1,
                    value=3
                )
            ], style={'columns': 2}),

            html.Label("Points per segment"),
            html.Div([
                dcc.Slider(
                    id='segment_len',
                    min=1,
                    max=5000,
                    value=256,
                    step=1),
                dcc.Input(
                    id='segment_len_input',
                    type='number',
                    min=1,
                    step=1,
                    value=256
                )
            ], style={'columns': 2}),

            html.Label(id='inspection'),
            html.Div([
                dcc.Graph(id='coherence_graph', style={'width': '100%', 'height': '100%'}),
                html.Hr(),
                html.Button('Export coherence', id='pick_coherence'),
                html.A('Export coherence',
                       id='link-coherence',
                       download="coherence.txt",
                       href="",
                       target="_blank",
                       hidden=True,
                       style={'textAlign': 'right'}),
                dcc.Checklist(
                    id='all_coherence',
                    options=[{'label': 'all signals', 'value': 'ALL'}]
                    ),

                html.Div([
                    html.Label("Resize graph"),
                    dcc.Slider(
                        id='graph_width2',
                        min=1,
                        max=15,
                        value=10,
                        marks={str(i): str(i) for i in range(1, 16)},
                        step=None),
                    dcc.Slider(
                        id='graph_height2',
                        min=1,
                        max=15,
                        value=8,
                        marks={str(i): str(i) for i in range(1, 16)},
                        step=None)
                ], style={'width': '40%'})
            ])

        ]),
        dcc.Tab(label='Schematisation', children=[
            html.Div([
                dcc.Dropdown(
                    id='schematisation'
                )
            ], style={'display': 'inline-block', 'width': '20%'}),

            html.Div([
                dcc.Checklist(
                    id='schem_sigs_prepare',
                    options=[
                        {'label': 'merged', 'value': 'MG'},
                    ],
                    value=['MG', 'EX'],
                    labelStyle={'display': 'inline-block'}
                    ),
                dcc.RadioItems(
                    id='schem_filter',
                    options=[
                        {'label': 'input signal', 'value': 'RW'},
                        {'label': 'smoothing and centering', 'value': 'SM'},
                    ],
                    value='SM',
                    labelStyle={'display': 'inline-block'}
                ),
            ]),


            dcc.Checklist(
                id='schem_sigs',
                options=[
                    {'label': 'show signal', 'value': 'SG'},
                    {'label': 'show merged', 'value': 'MG'},
                    {'label': 'show extremes', 'value': 'EX'}
                ],
                value=['MG', 'EX'],
                labelStyle={'display': 'inline-block'}
            ),
            dcc.RadioItems(
                id='schem_lines',
                options=[
                    {'label': 'lines', 'value': 'LL'},
                    {'label': 'lines+markers', 'value': 'LM'}
                ],
                value='LM',
                labelStyle={'display': 'inline-block'}
            ),
            html.Label("Smoothing window size"),
            html.Div([
                dcc.Slider(
                    id='smoothing_window_schem',
                    min=1,
                    max=300,
                    value=3,
                    step=1),
                dcc.Input(
                    id='smoothing_window_input_schem',
                    type='number',
                    min=1,
                    max=300,
                    step=1,
                    value=3
                )
            ], style={'columns': 2}),
            html.Label("Amplitude filter"),
            html.Div([
                dcc.Slider(
                    id='amplitude_width',
                    marks={str(i): mark(i) for i in range(1, 51)},
                    min=1,
                    max=50,
                    value=2,
                    step=None),
                dcc.Input(
                    id='amplitude_width_input',
                    type='number'
                )
            ], style={'columns': 2}),

            html.Label("Input statistics", id='input_stats'),

            html.Div([
                dcc.Graph(id='schem_graph', style={'width': '100%', 'height': '100%'}),
                html.Hr(),

                html.Button('Export cycles', id='pick_cycles'),
                html.A('Export cycles',
                       id='link-cycles',
                       download="cycles.txt",
                       href="",
                       target="_blank",
                       hidden=True,
                       style={'textAlign': 'right'}),

                html.Div([
                    html.Label("Resize graph"),
                    dcc.Slider(
                        id='graph_width3',
                        min=1,
                        max=15,
                        value=10,
                        marks={str(i): str(i) for i in range(1, 16)},
                        step=None),
                    dcc.Slider(
                        id='graph_height3',
                        min=1,
                        max=15,
                        value=8,
                        marks={str(i): str(i) for i in range(1, 16)},
                        step=None)
                ], style={'width': '40%'})
            ]),

            html.Hr(),
            html.Label("Correlation Table"),
            dcc.RadioItems(
                id='corr_table_code',
                options=[
                    {'label': 'min/max', 'value': 'MM'},
                    {'label': 'mean/range', 'value': 'MR'}
                ],
                value='MM',
                labelStyle={'display': 'inline-block'}
            ),
            html.Label("Classes range"),
            html.Div([
                dcc.Input(
                    id='class_min_input',
                    type='number',
                    step=0.01
                ),
                dcc.Input(
                    id='class_max_input',
                    type='number',
                    step=0.01
                ),
            ], style={'columns': 2}),
            html.Label("Classes count"),
            html.Div([
                dcc.Slider(
                    id='class_number',
                    marks={str(i): str(i) for i in range(1, 101)},
                    value=10),
                dcc.Input(
                    id='class_number_input',
                    type='number',
                    value=10
                )
            ], style={'columns': 2}),

            html.Div([
                dcc.Graph(id='table_map', style={'width': '100%', 'height': '100%'}),
                html.Hr(),
                html.Button('Export table', id='pick_table'),
                html.A('Export table',
                       id='link-table',
                       download="corr_table.txt",
                       href="",
                       target="_blank",
                       hidden=True,
                       style={'textAlign': 'right'}),
                html.Hr(),
                html.Div([
                    html.Label("Resize graph"),
                    dcc.Slider(
                        id='graph_width4',
                        min=1,
                        max=15,
                        value=5,
                        marks={str(i): str(i) for i in range(1, 16)},
                        step=None),
                    dcc.Slider(
                        id='graph_height4',
                        min=1,
                        max=15,
                        value=5,
                        marks={str(i): str(i) for i in range(1, 16)},
                        step=None)
                ], style={'width': '40%'})
            ])

        ])
    ], style={'height': 60}),

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


@app.callback(Output('loading_data', 'children'),
              Output('signal_1', 'options'),
              Output('filename', 'children'),
              Output('time_range', 'children'),
              Output('spectrum_1', 'options'),
              Output('spectrum_2', 'options'),
              Output('coherence_1', 'options'),
              Output('coherence_2', 'options'),
              Output('schematisation', 'options'),
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
    trp = [None, None, None]
    if contents:
        df = parse_data(contents, filename)
        cols = df.columns
        trp = prepare.calc_time_range(df[cols[0]].to_numpy())
        time_range = f"Time from {trp[0]} to {trp[1]}, mean time step is {trp[2]:.3e}, " \
                     f"time step deviation is {trp[3]:.3e}"
    options = get_options(df.columns)

    return [df.to_json(date_format='iso', orient='split'),
            options[1:], filename, time_range,
            options[1:], options[1:],
            options[1:], options[1:],
            options[1:],
            trp[0], trp[1], trp[2],
            trp[0], trp[1], trp[2],
            trp[0], trp[1], trp[2]]


@app.callback(Output('smoothing_window', 'value'),
              Output('smoothing_window_input', 'value'),
              Input('smoothing_window', 'value'),
              Input('smoothing_window_input', 'value')
              )
def set_smoothing_window(sldr, inpt):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    print(trigger_id)
    new_slider = sldr
    new_input = inpt

    if trigger_id == 'smoothing_window':
        new_input = sldr
    if trigger_id == 'smoothing_window_input':
        new_slider = inpt

    return [new_slider, new_input]


@app.callback(Output('smoothing_window_spectrum', 'value'),
              Output('smoothing_window_input_spectrum', 'value'),
              Input('smoothing_window_spectrum', 'value'),
              Input('smoothing_window_input_spectrum', 'value')
              )
def set_smoothing_window(sldr, inpt):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    print(trigger_id)
    new_slider = sldr
    new_input = inpt

    if trigger_id == 'smoothing_window_spectrum':
        new_input = sldr
    if trigger_id == 'smoothing_window_input_spectrum':
        new_slider = inpt

    return [new_slider, new_input]


@app.callback(Output('smoothing_window_coherence', 'value'),
              Output('smoothing_window_input_coherence', 'value'),
              Input('smoothing_window_coherence', 'value'),
              Input('smoothing_window_input_coherence', 'value')
              )
def set_smoothing_window(sldr, inpt):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    print(trigger_id)
    new_slider = sldr
    new_input = inpt

    if trigger_id == 'smoothing_window_coherence':
        new_input = sldr
    if trigger_id == 'smoothing_window_input_coherence':
        new_slider = inpt

    return [new_slider, new_input]


@app.callback(Output('segment_len', 'value'),
              Output('segment_len_input', 'value'),
              Input('segment_len', 'value'),
              Input('segment_len_input', 'value')
              )
def set_smoothing_window(sldr, inpt):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    print(trigger_id)
    new_slider = sldr
    new_input = inpt

    if trigger_id == 'segment_len':
        new_input = sldr
    if trigger_id == 'segment_len_input':
        new_slider = inpt

    return [new_slider, new_input]


@app.callback(Output('smoothing_window_schem', 'value'),
              Output('smoothing_window_input_schem', 'value'),
              Input('smoothing_window_schem', 'value'),
              Input('smoothing_window_input_schem', 'value')
              )
def set_smoothing_window(sldr, inpt):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    print(trigger_id)
    new_slider = sldr
    new_input = inpt

    if trigger_id == 'smoothing_window_schem':
        new_input = sldr
    if trigger_id == 'smoothing_window_input_schem':
        new_slider = inpt

    return [new_slider, new_input]


@app.callback(Output('class_number', 'value'),
              Output('class_number_input', 'value'),
              Input('class_number', 'value'),
              Input('class_number_input', 'value')
              )
def set_class_number(sldr, inpt):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    new_slider = sldr
    new_input = inpt

    if trigger_id == 'class_number':
        new_input = sldr
    if trigger_id == 'class_number_input':
        new_slider = inpt

    return [new_slider, new_input]


@app.callback(Output('smoothing_window', 'max'),
              Output('smoothing_window_input', 'max'),
              Output('smoothing_window_spectrum', 'max'),
              Output('smoothing_window_input_spectrum', 'max'),
              Output('smoothing_window_coherence', 'max'),
              Output('smoothing_window_input_coherence', 'max'),
              Output('segment_len', 'max'),
              Output('segment_len_input', 'max'),
              Output('smoothing_window_schem', 'max'),
              Output('smoothing_window_input_schem', 'max'),
              Input('t_start', 'value'),
              Input('t_end', 'value'),
              State('t_start', 'min'),
              State('t_start', 'max'),
              State('t_start', 'step')
              )
def update_smoothing_windows(t_start, t_end, t_min, t_max, t_step):
    # set time range if None
    val1 = t_min if t_start is None else t_start
    val2 = t_max if t_end is None else t_end
    step = 1.0 if t_step is None else t_step
    mx = 256
    if not (None in [val1, val2, t_step]):
        mx = (val2 - val1) / step
    mx2 = int(mx / 2)
    return [mx2, mx2, mx2, mx2, mx2, mx2, mx, mx, mx2, mx2]


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
              Input('graph_lines', 'value'),
              State('loading_data', 'children'),
              State('t_start', 'min'),
              State('t_start', 'max'),
              State('t_start', 'step'))
def update_graph(signal_1, signal_filter, k, graph_width, graph_height,
                 t_start, t_end, t_range, mode, loading_data, t_min, t_max, t_step):
    # set time range if None
    val1 = t_min if t_start is None else t_start
    val2 = t_max if t_end is None else t_end
    dt = 0.0 if t_step is None else t_step / 2

    # update time range if it changes
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if trigger_id == "t_start" or trigger_id == 't_end':
        val1 = t_start
        val2 = t_end
    if trigger_id == "time_range_slider":
        val1 = t_range[0]
        val2 = t_range[1]

    gmode = 'lines+markers' if mode == 'LM' else 'lines'

    data = []
    if signal_1:
        df = pd.read_json(loading_data, orient='split')
        cols = df.columns
        dff = df[(df[cols[0]] >= (val1 - dt)) & (df[cols[0]] <= (val2 + dt))]
        dff.reset_index(drop=True, inplace=True)
        for yy in signal_1:
            sig = dff[[cols[0], yy]]
            print(sig)

            data.append(go.Scatter(x=sig[cols[0]], y=sig[yy], mode=gmode, name=yy))

            if 'SM' in signal_filter:
                sig = prepare.smoothing_symm(sig, yy, k, 1)
                data.append(go.Scatter(x=sig[cols[0]], y=sig[yy], mode=gmode, name='smooth'))

            if 'HW' in signal_filter:
                sig = prepare.correction_hann(sig, yy)
                data.append(go.Scatter(x=sig[cols[0]], y=sig[yy], mode=gmode, name='hann_correction'))

    layout = go.Layout(xaxis={'title': 'Time'},
                       yaxis={'title': 'Input'},
                       margin={'l': 40, 'b': 40, 't': 50, 'r': 50},
                       hovermode='closest',
                       width=150 * graph_width, height=100 * graph_height)

    fig = go.Figure(data=data, layout=layout)

    return [fig, [val1, val2], val1, val2]


@app.callback(Output('link-signals', 'href'),
              Output('link-signals', 'hidden'),
              Input('all_signal', 'value'),
              Input('pick_signals', 'n_clicks'),
              Input('signal_1', 'value'),
              Input('signal_filter', 'value'),
              Input('smoothing_window', 'value'),
              Input('t_start', 'value'),
              Input('t_end', 'value'),
              State('loading_data', 'children'),
              State('t_start', 'min'),
              State('t_start', 'max'),
              State('t_start', 'step'))
def export_signal(all_check, n_clicks, yy, signal_filter, smoothing,
                  t_start, t_end, loading_data, t_min, t_max, t_step):
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    # button click
    if triggered_id == 'pick_signals':
        print("export signals")
        dff = pd.DataFrame()
        # set time range if None
        val1 = t_min if t_start is None else t_start
        val2 = t_max if t_end is None else t_end
        dt = 0.0 if t_step is None else t_step / 2
        if not (loading_data is None):
            df = pd.read_json(loading_data, orient='split')
            if not df.empty:
                cols = df.columns
                dff = df[(df[cols[0]] >= (val1 - dt)) & (df[cols[0]] <= (val2 + dt))]
                dff.reset_index(drop=True, inplace=True)
                export_cols = cols
                if all_check is None or all_check == []:
                    dff = dff[[cols[0]] + yy]
                    export_cols = yy
                # request smoothing signals
                if 'SM' in signal_filter:
                    dff = prepare.set_smoothing_symm(dff, export_cols, smoothing, 1)
                if 'HW' in signal_filter:
                    dff = prepare.set_correction_hann(dff, export_cols)

        csv_string = dff.to_csv(index=False, encoding='utf-8')
        csv_string = "data:text/csv;charset=utf-8,%EF%BB%BF" + urllib.parse.quote(csv_string)
        print("string={}".format(csv_string))
        return [csv_string, False]
    # changing
    return ["data:text/csv;charset=utf-8,%EF%BB%BF", True]


@app.callback(Output('spectrum_graph', 'figure'),
              Input('spectrum_1', 'value'),
              Input('spectrum_2', 'value'),
              Input('spectrum_filter', 'value'),
              Input('smoothing_window_spectrum', 'value'),
              Input('graph_width1', 'value'),
              Input('graph_height1', 'value'),
              Input('spectrum_lines', 'value'),
              State('t_start', 'value'),
              State('t_end', 'value'),
              State('t_start', 'step'),
              State('loading_data', 'children'))
def update_graph(spectrum_1, spectrum_2, spectrum_filter, k, graph_width, graph_height, mode,
                 t_start, t_end, t_step, loading_data):
    if spectrum_1 == spectrum_2:
        gname = 'power spectral density'
        fig = make_subplots(rows=1, cols=1)
    else:
        gname = 'cross-spectrum'
        fig = make_subplots(rows=2, cols=1)
    gmode = 'lines+markers' if mode == 'LM' else 'lines'

    if spectrum_1 and spectrum_2:
        df = pd.read_json(loading_data, orient='split')
        cols = df.columns
        val1 = df[cols[0]].iloc[0] if t_start is None else t_start
        val2 = df[cols[0]].iloc[-1] if t_end is None else t_end
        dt = 0.0 if t_step is None else t_step / 2
        dff = df[(df[cols[0]] >= (val1 - dt)) & (df[cols[0]] <= (val2 + dt))]
        dff.reset_index(drop=True, inplace=True)
        sig1 = dff[[cols[0], spectrum_1]]
        sig2 = dff[[cols[0], spectrum_2]]
        hann_koef = 1.0

        if 'SM' in spectrum_filter:
            sig1 = prepare.smoothing_symm(sig1, spectrum_1, k, 1)
            sig2 = prepare.smoothing_symm(sig2, spectrum_2, k, 1)

        if 'HW' in spectrum_filter:
            sig1 = prepare.correction_hann(sig1, spectrum_1)
            sig2 = prepare.correction_hann(sig2, spectrum_2)
            hann_koef = 8.0 / 3

        trp = prepare.calc_time_range(df[cols[0]].to_numpy())
        f, g_xy = signal.csd(sig1[spectrum_1], sig2[spectrum_2], (1.0 / trp[2]), window="boxcar", nperseg=len(sig1))
        mod, phase = analyse.cross_spectrum_mod_fas(g_xy)
        mod *= hann_koef

        fig.add_trace(go.Scatter(x=f, y=mod, mode=gmode, name=gname), row=1, col=1)
        if gname == 'cross-spectrum':
            fig.add_trace(go.Scatter(x=f, y=phase, mode=gmode, name='phase'), row=2, col=1)
    if gname == 'cross-spectrum':
        fig.update_xaxes(title_text="Frequencies", row=1, col=1)
        fig.update_xaxes(title_text="Frequencies", row=2, col=1)
        fig.update_yaxes(title_text="Cross Spectrum Module", row=1, col=1)
        fig.update_yaxes(title_text="Cross Spectrum Phase", row=2, col=1)
    else:
        fig.update_xaxes(title_text="Frequencies", row=1, col=1)
        fig.update_yaxes(title_text="Power Spectral Density", row=1, col=1)
    fig.update_layout(width=150 * graph_width, height=100 * graph_height)
    return fig


@app.callback(Output('schem_graph', 'figure'),
              Input('schematisation', 'value'),
              Input('schem_filter', 'value'),
              Input('schem_sigs', 'value'),
              Input('schem_sigs_prepare', 'value'),
              Input('smoothing_window_schem', 'value'),
              Input('graph_width3', 'value'),
              Input('graph_height3', 'value'),
              Input('spectrum_lines', 'value'),
              Input('amplitude_width_input', 'value'),
              State('t_start', 'value'),
              State('t_end', 'value'),
              State('t_start', 'step'),
              State('loading_data', 'children'))
def update_graph(signal1, schem_filter, schem_sigs, is_merged, k, graph_width, graph_height, mode, eps,
                 t_start, t_end, t_step, loading_data):
    gmode = 'lines+markers' if mode == 'LM' else 'lines'
    data = []
    if signal1:
        df = pd.read_json(loading_data, orient='split')
        cols = df.columns
        val1 = df[cols[0]].iloc[0] if t_start is None else t_start
        val2 = df[cols[0]].iloc[-1] if t_end is None else t_end
        dt = 0.0 if t_step is None else t_step / 2
        dff = df[(df[cols[0]] >= (val1 - dt)) & (df[cols[0]] <= (val2 + dt))]
        dff.reset_index(drop=True, inplace=True)
        sig = dff[[cols[0], signal1]]
        if schem_filter == 'SM':
            sig = prepare.smoothing_symm(sig, signal1, k, 1)

        if 'SG' in schem_sigs:
            data.append(go.Scatter(x=sig[cols[0]], y=sig[signal1], mode=gmode, name='input'))

        if 'MG' in schem_sigs:
            sig = schematisation.merge(sig, signal1, eps)
            data.append(go.Scatter(x=sig[cols[0]], y=sig[signal1], mode=gmode, name='merge'))

        if 'EX' in schem_sigs:
            if 'MG' in is_merged:
                sig = schematisation.merge(sig, signal1, eps)
            sig = schematisation.pick_extremes(sig, signal1)
            data.append(go.Scatter(x=sig[cols[0]], y=sig[signal1], mode=gmode, name='extremes'))

    layout = go.Layout(xaxis={'title': 'Time'},
                       yaxis={'title': 'Input'},
                       margin={'l': 40, 'b': 40, 't': 50, 'r': 50},
                       hovermode='closest',
                       width=150 * graph_width, height=100 * graph_height)

    fig = go.Figure(data=data, layout=layout)
    return fig


@app.callback(Output('table_map', 'figure'),
              Input('schematisation', 'value'),
              Input('schem_filter', 'value'),
              Input('schem_sigs', 'value'),
              Input('smoothing_window_schem', 'value'),
              Input('graph_width4', 'value'),
              Input('graph_height4', 'value'),
              Input('amplitude_width_input', 'value'),
              Input('class_min_input', 'value'),
              Input('class_max_input', 'value'),
              Input('class_number', 'value'),
              Input('corr_table_code', 'value'),
              State('t_start', 'value'),
              State('t_end', 'value'),
              State('t_start', 'step'),
              State('loading_data', 'children'))
def update_graph(signal1, schem_filter, schem_sigs, k, graph_width, graph_height, eps, class_min, class_max, m, code,
                 t_start, t_end, t_step, loading_data):
    tbl = pd.DataFrame()
    x_title = ''
    y_title = ''
    if signal1:
        df = pd.read_json(loading_data, orient='split')
        cols = df.columns
        val1 = df[cols[0]].iloc[0] if t_start is None else t_start
        val2 = df[cols[0]].iloc[-1] if t_end is None else t_end
        dt = 0.0 if t_step is None else t_step / 2
        dff = df[(df[cols[0]] >= (val1 - dt)) & (df[cols[0]] <= (val2 + dt))]
        dff.reset_index(drop=True, inplace=True)
        sig = dff[[cols[0], signal1]]
        if schem_filter == 'SM':
            sig = prepare.smoothing_symm(sig, signal1, k, 1)

        if 'MG' in schem_sigs:
            sig = schematisation.merge(sig, signal1, eps)

        sig = schematisation.pick_extremes(sig, signal1)
        # print(sig)
        cycles = schematisation.pick_cycles_as_df(sig, signal1)
        # print(cycles)

        if code == 'MM':
            tbl = schematisation.correlation_table(cycles, 'Max', 'Min', class_min, class_max, m)
            x_title = 'Min'
            y_title = 'Max'
        if code == 'MR':
            tbl = schematisation.correlation_table(cycles, 'Range', 'Mean', class_min, class_max, m)
            x_title = 'Mean'
            y_title = 'Range'

    fig = px.imshow(tbl, color_continuous_scale='GnBu')
    fig.update_layout(width=150 * graph_width, height=100 * graph_height, margin=dict(l=10, r=10, b=10, t=10),
                      xaxis={'title': x_title}, yaxis={'title': y_title})
    fig.update_xaxes(side="top")
    return fig


@app.callback(Output('link-cycles', 'href'),
              Output('link-cycles', 'hidden'),
              Input('pick_cycles', 'n_clicks'),
              Input('schematisation', 'value'),
              Input('schem_filter', 'value'),
              Input('schem_sigs', 'value'),
              Input('smoothing_window_schem', 'value'),
              Input('amplitude_width_input', 'value'),
              State('t_start', 'value'),
              State('t_end', 'value'),
              State('t_start', 'step'),
              State('loading_data', 'children'))
def export_cycles(n_clicks, signal1, schem_filter, schem_sigs, k, eps, t_start, t_end, t_step, loading_data):
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    # button click
    if triggered_id == 'pick_cycles':
        sig = pd.DataFrame()
        if signal1:
            df = pd.read_json(loading_data, orient='split')
            cols = df.columns
            val1 = df[cols[0]].iloc[0] if t_start is None else t_start
            val2 = df[cols[0]].iloc[-1] if t_end is None else t_end
            dt = 0.0 if t_step is None else t_step / 2
            dff = df[(df[cols[0]] >= (val1 - dt)) & (df[cols[0]] <= (val2 + dt))]
            dff.reset_index(drop=True, inplace=True)
            sig = dff[[cols[0], signal1]]
            if schem_filter == 'SM':
                sig = prepare.smoothing_symm(sig, signal1, k, 1)

            if 'MG' in schem_sigs:
                sig = schematisation.merge(sig, signal1, eps)

            sig = schematisation.pick_extremes(sig, signal1)
            sig = schematisation.pick_cycles_as_df(sig, signal1)
        csv_string = sig.to_csv(index=False, encoding='utf-8')
        csv_string = "data:text/csv;charset=utf-8,%EF%BB%BF" + urllib.parse.quote(csv_string)
        return [csv_string, False]
    # change something
    return ["data:text/csv;charset=utf-8,%EF%BB%BF", True]


@app.callback(Output('link-table', 'href'),
              Output('link-table', 'hidden'),
              Input('pick_table', 'n_clicks'),
              Input('schematisation', 'value'),
              Input('schem_filter', 'value'),
              Input('schem_sigs', 'value'),
              Input('smoothing_window_schem', 'value'),
              Input('amplitude_width_input', 'value'),
              Input('class_min_input', 'value'),
              Input('class_max_input', 'value'),
              Input('class_number', 'value'),
              Input('corr_table_code', 'value'),
              State('t_start', 'value'),
              State('t_end', 'value'),
              State('t_start', 'step'),
              State('loading_data', 'children'))
def export_table(n_clicks, signal1, schem_filter, schem_sigs, k, eps, class_min, class_max, m, code,
                 t_start, t_end, t_step, loading_data):
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    # button click
    if triggered_id == 'pick_table':
        dff = pd.DataFrame()
        if signal1:
            df = pd.read_json(loading_data, orient='split')
            cols = df.columns
            val1 = df[cols[0]].iloc[0] if t_start is None else t_start
            val2 = df[cols[0]].iloc[-1] if t_end is None else t_end
            dt = 0.0 if t_step is None else t_step / 2
            dff = df[(df[cols[0]] >= (val1 - dt)) & (df[cols[0]] <= (val2 + dt))]
            dff.reset_index(drop=True, inplace=True)
            sig = dff[[cols[0], signal1]]
            if schem_filter == 'SM':
                sig = prepare.smoothing_symm(sig, signal1, k, 1)

            if 'MG' in schem_sigs:
                sig = schematisation.merge(sig, signal1, eps)

            sig = schematisation.pick_extremes(sig, signal1)
            cycles = schematisation.pick_cycles_as_df(sig, signal1)
            if code == 'MM':
                dff = schematisation.correlation_table(cycles, 'Max', 'Min', class_min, class_max, m)
            if code == 'MR':
                dff = schematisation.correlation_table(cycles, 'Range', 'Mean', class_min, class_max, m)

        csv_string = dff.to_csv(index=True, encoding='utf-8')
        csv_string = "data:text/csv;charset=utf-8,%EF%BB%BF" + urllib.parse.quote(csv_string)
        return [csv_string, False]
    # change something
    return ["data:text/csv;charset=utf-8,%EF%BB%BF", True]


@app.callback(Output('input_stats', 'children'),
              Input('schematisation', 'value'),
              State('t_start', 'value'),
              State('t_end', 'value'),
              State('t_start', 'step'),
              State('loading_data', 'children'))
def print_input_stats(signal1, t_start, t_end, t_step, loading_data):
    inp_str = ''
    if signal1:
        df = pd.read_json(loading_data, orient='split')
        cols = df.columns
        val1 = df[cols[0]].iloc[0] if t_start is None else t_start
        val2 = df[cols[0]].iloc[-1] if t_end is None else t_end
        dt = 0.0 if t_step is None else t_step / 2
        dff = df[(df[cols[0]] >= (val1 - dt)) & (df[cols[0]] <= (val2 + dt))]
        dff.reset_index(drop=True, inplace=True)
        s_mean, s_variance, s_deviation, s_koef = schematisation.input_stats(dff, signal1)
        inp_str = 'Mean value is {:.3e}, standard deviation is {:.3e}, ' \
                  'irregular coefficient is {:.3e}'.format(s_mean, s_deviation, s_koef)
    return inp_str


@app.callback(Output('amplitude_width', 'value'),
              Output('amplitude_width_input', 'value'),
              Output('amplitude_width_input', 'max'),
              Output('schem_sigs', 'value'),
              Input('amplitude_width', 'value'),
              Input('amplitude_width_input', 'value'),
              Input('schematisation', 'value'),
              State('t_start', 'value'),
              State('t_end', 'value'),
              State('t_start', 'min'),
              State('t_start', 'max'),
              State('t_start', 'step'),
              State('amplitude_width_input', 'max'),
              State('loading_data', 'children'),
              State('schem_sigs', 'value'))
def amplitude_filter(sldr, inpt, signal1, t_start, t_end, t_min, t_max, t_step, cur_max, loading_data, current_sigs):
    # set time range if None
    val1 = t_min if t_start is None else t_start
    val2 = t_max if t_end is None else t_end
    dt = 0.0 if t_step is None else t_step / 2

    # update amplitude filter and input value if signal changed
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    current_slider = sldr
    current_inpt = inpt
    current_range = cur_max

    if trigger_id == 'schematisation':
        if signal1:
            df = pd.read_json(loading_data, orient='split')
            cols = df.columns
            dff = df[(df[cols[0]] >= (val1 - dt)) & (df[cols[0]] <= (val2 + dt))]
            dff.reset_index(drop=True, inplace=True)
            current_range = dff[signal1].max() - dff[signal1].min()
            current_inpt = current_slider * current_range / 100.0

    if trigger_id == 'amplitude_width':
        current_sigs += ['MG']
        if signal1:
            current_inpt = current_slider * current_range / 100.0

    if trigger_id == 'amplitude_width_input':
        current_sigs += ['MG']
        if signal1:
            current_slider = current_inpt / current_range * 100

    print('sldr={}, inpt={}, range={}'.format(current_slider, current_inpt, current_range))

    return [current_slider, current_inpt, current_range, current_sigs]


@app.callback(Output('link-spectrum', 'href'),
              Output('link-spectrum', 'hidden'),
              Input('all_spectrum', 'value'),
              Input('pick_spectrum', 'n_clicks'),
              Input('spectrum_1', 'value'),
              Input('spectrum_2', 'value'),
              Input('t_start', 'value'),
              Input('t_end', 'value'),
              Input('spectrum_filter', 'value'),
              Input('smoothing_window_spectrum', 'value'),
              State('loading_data', 'children'),
              State('t_start', 'min'),
              State('t_start', 'max'),
              State('t_start', 'step'))
def export_cross_spectrum(all_checked, n_clicks, spectrum_1, spectrum_2, t_start, t_end, spectrum_filter, k,
                          loading_data, t_min, t_max, t_step):
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    # button click
    if triggered_id == 'pick_spectrum':
        print("in export cross")
        dfres = pd.DataFrame()
        # set time range if None
        val1 = t_min if t_start is None else t_start
        val2 = t_max if t_end is None else t_end
        dt = 0.0 if t_step is None else t_step / 2
        smoothing = k if 'SM' in spectrum_filter else -1
        win = "hann" if 'HW' in spectrum_filter else "boxcar"
        if not (loading_data is None):
            df = pd.read_json(loading_data, orient='split')
            if not df.empty:
                cols = df.columns
                dff = df[(df[cols[0]] >= (val1 - dt)) & (df[cols[0]] <= (val2 + dt))]
                dff.reset_index(drop=True, inplace=True)
                if all_checked is None or all_checked == []:
                    dfres = pipeline.calc_signals_cross_spectrum(dff, spectrum_1, spectrum_2, smoothing, win)
                else:
                    dfres = pipeline.calc_set_of_signals_cross_spectrum(dff, smoothing, win)
        csv_string = dfres.to_csv(index=False, encoding='utf-8')
        csv_string = "data:text/csv;charset=utf-8,%EF%BB%BF" + urllib.parse.quote(csv_string)
        return [csv_string, False]
    return ["data:text/csv;charset=utf-8,%EF%BB%BF", True]


@app.callback(Output('coherence_graph', 'figure'),
              # Output('inspection', 'children'),
              Input('coherence_1', 'value'),
              Input('coherence_2', 'value'),
              Input('coherence_filter', 'value'),
              Input('smoothing_window_coherence', 'value'),
              Input('segment_len', 'value'),
              Input('graph_width2', 'value'),
              Input('graph_height2', 'value'),
              Input('coherence_lines', 'value'),
              State('t_start', 'value'),
              State('t_end', 'value'),
              State('t_start', 'step'),
              State('loading_data', 'children'))
def update_graph(coherence_1, coherence_2, coherence_filter, k, segment_len, graph_width, graph_height,
                 mode, t_start, t_end, t_step, loading_data):
    data = []
    if coherence_1 and coherence_2:
        gmode = 'lines+markers' if mode == 'LM' else 'lines'
        df = pd.read_json(loading_data, orient='split')
        cols = df.columns
        val1 = df[cols[0]].iloc[0] if t_start is None else t_start
        val2 = df[cols[0]].iloc[-1] if t_end is None else t_end
        dt = 0.0 if t_step is None else t_step / 2
        dff = df[(df[cols[0]] >= (val1 - dt)) & (df[cols[0]] <= (val2 + dt))]
        dff.reset_index(drop=True, inplace=True)
        sig1 = dff[[cols[0], coherence_1]]
        sig2 = dff[[cols[0], coherence_2]]

        if 'SM' in coherence_filter:
            sig1 = prepare.smoothing_symm(sig1, coherence_1, k, 1)
            sig2 = prepare.smoothing_symm(sig2, coherence_2, k, 1)

        if 'HW' in coherence_filter:
            sig1 = prepare.correction_hann(sig1, coherence_1)
            sig2 = prepare.correction_hann(sig2, coherence_2)

        trp = prepare.calc_time_range(df[cols[0]].to_numpy())
        f, c_xx = signal.coherence(sig1[coherence_1], sig2[coherence_2],
                                   (1.0 / trp[2]), window="boxcar", nperseg=segment_len)

        data.append(go.Scatter(x=f, y=c_xx, mode=gmode, name='coherence'))

    layout = go.Layout(xaxis={'title': 'Frequencies'},
                       yaxis={'title': 'Coherence'},
                       margin={'l': 40, 'b': 40, 't': 50, 'r': 50},
                       hovermode='closest',
                       width=150 * graph_width, height=100 * graph_height)

    fig = go.Figure(data=data, layout=layout)

    return fig


@app.callback(Output('link-coherence', 'href'),
              Output('link-coherence', 'hidden'),
              Input('all_coherence', 'value'),
              Input('pick_coherence', 'n_clicks'),
              Input('coherence_1', 'value'),
              Input('coherence_2', 'value'),
              Input('t_start', 'value'),
              Input('t_end', 'value'),
              Input('coherence_filter', 'value'),
              Input('smoothing_window_coherence', 'value'),
              Input('segment_len', 'value'),
              State('loading_data', 'children'),
              State('t_start', 'min'),
              State('t_start', 'max'),
              State('t_start', 'step'))
def export_coherence(all_checked, n_clicks, coherence_1, coherence_2, t_start, t_end, spectrum_filter,
                     k, npseg, loading_data, t_min, t_max, t_step):
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    print(all_checked)
    # button click
    if triggered_id == 'pick_coherence':
        dfres = pd.DataFrame()
        # set time range if None
        val1 = t_min if t_start is None else t_start
        val2 = t_max if t_end is None else t_end
        dt = 0.0 if t_step is None else t_step / 2
        smoothing = k if 'SM' in spectrum_filter else -1
        win = "hann" if 'HW' in spectrum_filter else "boxcar"
        if loading_data:
            df = pd.read_json(loading_data, orient='split')
            if not df.empty:
                cols = df.columns
                dff = df[(df[cols[0]] >= (val1 - dt)) & (df[cols[0]] <= (val2 + dt))]
                dff.reset_index(drop=True, inplace=True)
                if all_checked is None or all_checked == []:
                    dfres = pipeline.calc_signals_coherence(dff, coherence_1, coherence_2, smoothing, win, npseg)
                else:
                    dfres = pipeline.calc_set_of_signals_coherence(dff, smoothing, win, npseg)
        csv_string = dfres.to_csv(index=False, encoding='utf-8')
        csv_string = "data:text/csv;charset=utf-8,%EF%BB%BF" + urllib.parse.quote(csv_string)
        return [csv_string, False]
    return ["data:text/csv;charset=utf-8,%EF%BB%BF", True]


if __name__ == '__main__':
    app.run_server(debug=True)
