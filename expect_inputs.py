import base64
import io
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import numpy as np
from staff import prepare
from staff import analyse
from staff import schematisation
from staff import pipeline
from scipy import signal
from staff import loaddata
import urllib
import json
import os

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

    # html.Label(id='time_range'),

    dcc.Tabs([
        dcc.Tab(label='Expect signal', children=[
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

            html.Div([
                dcc.Dropdown(
                    id='signal_1',
                    multi=True
                )
            ], style={'display': 'inline-block', 'width': '20%'}),

            dcc.Checklist(
                id='signal_filter',
                options=[
                    {'label': 'smoothing', 'value': 'SM'},
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

                    # dcc.Input(
                    #    id='t_start',
                    #    type='number'
                    # ),
                    # dcc.Input(
                    #    id='t_end',
                    #    type='number'
                    # ),
                    html.Div(
                        dcc.Dropdown(
                            id='start_1',
                            value=None
                        ), style={'width': '20%'}),
                    html.Div(
                        dcc.Dropdown(
                            id='end_1',
                            value=None
                        ), style={'width': '20%'})

                ], style={'width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center'}),

                html.Div([
                    dcc.RangeSlider(
                        id='time_range_slider',
                        step=None,
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
                    {'label': 'smoothing', 'value': 'SM'},
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
            html.A('Export spectrum',
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
                    {'label': 'smoothing', 'value': 'SM'},
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
        dcc.Tab(label='Expect cycles', children=[
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
                    value=['MG'],
                    labelStyle={'display': 'inline-block'}
                ),
                dcc.RadioItems(
                    id='schem_filter',
                    options=[
                        {'label': 'input signal', 'value': 'RW'},
                        {'label': 'smoothing', 'value': 'SM'},
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
            html.Label("Intervals for frequency estimation"),
            html.Div([
                dcc.Slider(
                    id='frequency_est',
                    min=1,
                    max=50,
                    value=25,
                    step=1),
                dcc.Input(
                    id='frequency_est_input',
                    type='number',
                    min=1,
                    max=50,
                    step=1,
                    value=25
                )
            ], style={'columns': 2}),
            html.Label("Max frequency", id='f_max'),

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
            ])

        ]),
        dcc.Tab(label='Correlation', children=[
            dcc.RadioItems(
                id='corr_table_code',
                options=[
                    {'label': 'min/max', 'value': 'MM'},
                    {'label': 'mean/range', 'value': 'MR'}
                ],
                value='MR',
                labelStyle={'display': 'inline-block'}
            ),
            html.Label("Classes intervals"),
            html.Label(id='range1', children='mean'),
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
            html.Label(id='range2', children='range'),
            html.Div([
                dcc.Input(
                    id='class_min_input2',
                    type='number',
                    step=0.01
                ),
                dcc.Input(
                    id='class_max_input2',
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
                html.Div([
                    html.Label("Resize graph"),
                    dcc.Slider(
                        id='graph_width4',
                        min=1,
                        max=15,
                        value=4,
                        marks={str(i): str(i) for i in range(1, 16)},
                        step=None),
                    dcc.Slider(
                        id='graph_height4',
                        min=1,
                        max=15,
                        value=6,
                        marks={str(i): str(i) for i in range(1, 16)},
                        step=None)
                ], style={'width': '40%'})
            ]),

            html.Label('Connected parameters'),
            html.Div([
                dcc.Dropdown(
                    id='traces',
                    multi=True
                )
            ], style={'display': 'inline-block', 'width': '20%'}),
            html.Label("Maximum time interval to average connected parameters"),
            html.Div([
                dcc.Slider(
                    id='dt_max',
                    value=None,
                    max=5.0),
                dcc.Input(
                    id='dt_max_input',
                    type='number',
                    max=5.0,
                    value=None
                )
            ], style={'columns': 2}),

            html.Div([
                dcc.Graph(id='trace_map', style={'width': '100%', 'height': '100%'}),
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
                        id='graph_width6',
                        min=1,
                        max=15,
                        value=9,
                        marks={str(i): str(i) for i in range(1, 16)},
                        step=None),
                    dcc.Slider(
                        id='graph_height6',
                        min=1,
                        max=15,
                        value=6,
                        marks={str(i): str(i) for i in range(1, 16)},
                        step=None)
                ], style={'width': '40%'})
            ])

        ]),
        dcc.Tab(label='Distribution Estimate', children=[
            html.Div([
                dcc.Upload(
                    id='upload_tables',
                    children=html.Div([
                        html.A('Select Files')
                    ]),
                    multiple=True
                ),
                html.H5(id='filenames'),
            ]),
            html.Div([
                dcc.Dropdown(
                    id='estimation'
                )
            ], style={'display': 'inline-block', 'width': '20%'}),

            dcc.RadioItems(
                id='corr_code',
                options=[
                    {'label': 'min/max', 'value': 'MM'},
                    {'label': 'mean/range', 'value': 'MR'}
                ],
                value='MM',
                labelStyle={'display': 'inline-block'}
            ),
            html.Div([
                dcc.Graph(id='distribution', style={'width': '100%', 'height': '100%'}),
                html.Hr(),
                html.Label("Levels"),
                html.Div([
                    dcc.Slider(
                        id='cut1',
                        min=1,
                        max=50,
                        value=2,
                        step=1),
                    dcc.Input(
                        id='cut1_input',
                        type='number',
                        min=1,
                        max=50,
                        step=1,
                        value=2
                    )
                ], style={'columns': 2}),
                html.Div([
                    dcc.Slider(
                        id='cut2',
                        min=1,
                        max=50,
                        value=2,
                        step=1),
                    dcc.Input(
                        id='cut2_input',
                        type='number',
                        min=1,
                        max=50,
                        step=1,
                        value=2
                    )
                ], style={'columns': 2}),

                html.Hr(),
                html.Div([
                    html.Label("Resize graph"),
                    dcc.Slider(
                        id='graph_width5',
                        min=1,
                        max=15,
                        value=10,
                        marks={str(i): str(i) for i in range(1, 16)},
                        step=None),
                    dcc.Slider(
                        id='graph_height5',
                        min=1,
                        max=15,
                        value=6,
                        marks={str(i): str(i) for i in range(1, 16)},
                        step=None)
                ], style={'width': '40%'}),

                dcc.Graph(id='integral', style={'width': '100%', 'height': '100%'}),
                html.Button('Export', id='cdf'),
                html.A('Export',
                       id='link-cdf',
                       download="cdf.txt",
                       href="",
                       target="_blank",
                       hidden=True,
                       style={'textAlign': 'right'}),
                html.Hr(),
                html.Div([
                    html.Label("Resize graph"),
                    dcc.Slider(
                        id='graph_width7',
                        min=1,
                        max=15,
                        value=10,
                        marks={str(i): str(i) for i in range(1, 16)},
                        step=None),
                    dcc.Slider(
                        id='graph_height7',
                        min=1,
                        max=15,
                        value=6,
                        marks={str(i): str(i) for i in range(1, 16)},
                        step=None)
                ], style={'width': '40%'})



            ])
        ])

    ], style={'height': 60}),

    html.Div(id='loading_data', style={'display': 'none'}),
    html.Div(id='loading_corr', style={'display': 'none'})
])


def get_options(names):
    return [{'label': name, 'value': name} for name in names]


def get_options_key(data):
    return [{'label': data[i], 'value': i} for i in range(len(data))]


def get_marks(data):
    return {str(i): '' for i in range(len(data))}


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
              Output('traces', 'options'),
              Output('start_1', 'options'),
              Output('end_1', 'options'),
              Output('time_range_slider', 'min'),
              Output('time_range_slider', 'max'),
              Output('time_range_slider', 'marks'),
              Output('dt_max', 'step'),
              Output('dt_max_input', 'step'),
              Input('upload_data', 'contents'),
              State('upload_data', 'filename'))
def upload_file(contents, filename):
    df = pd.DataFrame()
    time_range = ""
    time = get_options_key([])
    marks = get_marks([])
    trp = [None, None, None]
    if contents:
        df = loaddata.parse_data(contents, filename)
        cols = df.columns
        tm = df[cols[0]].to_numpy()
        trp = prepare.calc_time_range(tm)
        time_range = f"Time from {trp[0]} to {trp[1]}, mean time step is {trp[2]:.3e}, " \
                     f"time step deviation is {trp[3]:.3e}"
        time = get_options_key(tm)
        marks = get_marks(tm)

    options = get_options(df.columns)
    return [df.to_json(date_format='iso', orient='split'),
            options[1:], filename, time_range,
            options[1:], options[1:],
            options[1:], options[1:],
            options[1:], options[1:],
            time, time,
            0, max(len(marks) - 1, 0), marks,
            trp[2], trp[2]]


@app.callback(Output('smoothing_window', 'value'),
              Output('smoothing_window_input', 'value'),
              Input('smoothing_window', 'value'),
              Input('smoothing_window_input', 'value')
              )
def set_smoothing_window(sldr, inpt):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
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


@app.callback(Output('frequency_est', 'value'),
              Output('frequency_est_input', 'value'),
              Input('frequency_est', 'value'),
              Input('frequency_est_input', 'value')
              )
def set_frequency_est(sldr, inpt):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    new_slider = sldr
    new_input = inpt

    if trigger_id == 'frequency_est':
        new_input = sldr
    if trigger_id == 'frequency_est_input':
        new_slider = inpt

    return [new_slider, new_input]


@app.callback(Output('dt_max', 'value'),
              Output('dt_max_input', 'value'),
              Input('dt_max', 'value'),
              Input('dt_max_input', 'value'))
def set_dt_max(sldr, inpt):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    new_slider = sldr
    new_input = inpt

    if trigger_id == 'dt_max':
        new_input = sldr
    if trigger_id == 'dt_max_input':
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
              Input('start_1', 'value'),
              Input('end_1', 'value'),
              Input('start_1', 'options')
              )
def update_smoothing_windows(t_start, t_end, time):
    ctx = dash.callback_context
    trigger_id, opt = ctx.triggered[0]["prop_id"].split(".")
    if opt == 'options':
        ind1 = 0
        ind2 = max(len(time) - 1, 0)
    else:
        ind1 = t_start
        ind2 = t_end

    mx = max(ind2 - ind1, 6)
    mx2 = int(mx / 2)
    return [mx2, mx2, mx2, mx2, mx2, mx2, mx, mx, mx2, mx2]


@app.callback(Output('input_graph', 'figure'),
              Output('time_range_slider', 'value'),
              Output('start_1', 'value'),
              Output('end_1', 'value'),
              Input('start_1', 'options'),
              Input('signal_1', 'value'),
              Input('signal_filter', 'value'),
              Input('smoothing_window', 'value'),
              Input('graph_width', 'value'),
              Input('graph_height', 'value'),
              Input('start_1', 'value'),
              Input('end_1', 'value'),
              Input('time_range_slider', 'value'),
              Input('graph_lines', 'value'),
              State('loading_data', 'children'))
def update_graph(time, signal_1, signal_filter, k, graph_width, graph_height,
                 t_start, t_end, t_range, mode, loading_data):
    # set time range if None
    val1 = 0 if (t_start == 0) or (t_start is None) else t_start
    val2 = max(len(time) - 1, 0) if (t_end == 0) or (t_end is None) else t_end

    # update time range if it changes
    ctx = dash.callback_context
    trigger_id, opt = ctx.triggered[0]["prop_id"].split(".")

    if trigger_id == "start_1" or trigger_id == 'end_1':
        val1 = t_start
        val2 = t_end
    if trigger_id == "time_range_slider":
        val1 = t_range[0]
        val2 = t_range[1]

    gmode = 'lines+markers' if mode == 'LM' else 'lines'

    data = []
    if signal_1:
        time1 = time[val1]['label']
        time2 = time[val2]['label']
        dff = loaddata.select_dff_by_time(loading_data, time1, time2, 0)
        cols = dff.columns
        for yy in signal_1:
            data.append(go.Scatter(x=dff[cols[0]], y=dff[yy], mode=gmode, name=yy))

            if 'SM' in signal_filter:
                dff = prepare.smoothing_symm(dff, yy, k, 1)
                data.append(go.Scatter(x=dff[cols[0]], y=dff[yy], mode=gmode, name='smooth'))

            if 'HW' in signal_filter:
                dff = prepare.correction_hann(dff, yy)
                data.append(go.Scatter(x=dff[cols[0]], y=dff[yy], mode=gmode, name='hann_correction'))

    layout = go.Layout(xaxis={'title': 'Time'},
                       yaxis={'title': 'Input'},
                       margin={'l': 40, 'b': 40, 't': 50, 'r': 50},
                       hovermode='closest',
                       width=150 * graph_width, height=100 * graph_height)

    fig = go.Figure(data=data, layout=layout)

    return [fig, [val1, val2], val1, val2]


@app.callback(Output('spectrum_graph', 'figure'),
              Input('start_1', 'options'),
              Input('spectrum_1', 'value'),
              Input('spectrum_2', 'value'),
              Input('spectrum_filter', 'value'),
              Input('smoothing_window_spectrum', 'value'),
              Input('graph_width1', 'value'),
              Input('graph_height1', 'value'),
              Input('spectrum_lines', 'value'),
              State('start_1', 'value'),
              State('end_1', 'value'),
              State('loading_data', 'children'))
def update_graph(time, spectrum_1, spectrum_2, spectrum_filter, k, graph_width, graph_height, mode,
                 t_start, t_end, loading_data):
    if spectrum_1 == spectrum_2:
        gname = 'power spectral density'
        fig = make_subplots(rows=1, cols=1)
    else:
        gname = 'cross-spectrum'
        fig = make_subplots(rows=2, cols=1)
    gmode = 'lines+markers' if mode == 'LM' else 'lines'

    if spectrum_1 and spectrum_2:
        val1 = 0 if (t_start == 0) or (t_start is None) else t_start
        val2 = max(len(time) - 1, 0) if (t_end == 0) or (t_end is None) else t_end
        time1 = time[val1]['label']
        time2 = time[val2]['label']
        dff = loaddata.select_dff_by_time(loading_data, time1, time2, 0)
        hann_koef = 1.0

        if 'SM' in spectrum_filter and not (k is None):
            dff = prepare.set_smoothing_symm(dff, [spectrum_1, spectrum_2], k, 1)

        if 'HW' in spectrum_filter:
            dff = prepare.set_correction_hann(dff, [spectrum_1, spectrum_2])
            hann_koef = 8.0 / 3

        rows, _ = dff.shape
        cols = dff.columns
        tm = dff[cols[0]].to_numpy()
        trp = prepare.calc_time_range(tm)
        f, g_xy = signal.csd(dff[spectrum_1], dff[spectrum_2], (1.0 / trp[2]), window="boxcar", nperseg=rows)
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


@app.callback(Output('coherence_graph', 'figure'),
              Input('coherence_1', 'value'),
              Input('coherence_2', 'value'),
              Input('coherence_filter', 'value'),
              Input('smoothing_window_coherence', 'value'),
              Input('segment_len', 'value'),
              Input('graph_width2', 'value'),
              Input('graph_height2', 'value'),
              Input('coherence_lines', 'value'),
              State('start_1', 'options'),
              State('start_1', 'value'),
              State('end_1', 'value'),
              State('loading_data', 'children'))
def update_graph(coherence_1, coherence_2, coherence_filter, k, segment_len, graph_width, graph_height,
                 mode, time, t_start, t_end, loading_data):
    data = []
    gmode = 'lines+markers' if mode == 'LM' else 'lines'
    if coherence_1 and coherence_2:
        val1 = 0 if (t_start == 0) or (t_start is None) else t_start
        val2 = max(len(time) - 1, 0) if (t_end == 0) or (t_end is None) else t_end
        time1 = time[val1]['label']
        time2 = time[val2]['label']

        dff = loaddata.select_dff_by_time(loading_data, time1, time2, 0)

        if 'SM' in coherence_filter and not (k is None):
            dff = prepare.set_smoothing_symm(dff, [coherence_1, coherence_2], k, 1)

        if 'HW' in coherence_filter:
            dff = prepare.set_correction_hann(dff, [coherence_1, coherence_2])

        cols = dff.columns
        tm = dff[cols[0]].to_numpy()
        trp = prepare.calc_time_range(tm)
        f, c_xx = signal.coherence(dff[coherence_1], dff[coherence_2],
                                   (1.0 / trp[2]), window="boxcar", nperseg=segment_len)

        data.append(go.Scatter(x=f, y=c_xx, mode=gmode, name='coherence'))

    layout = go.Layout(xaxis={'title': 'Frequencies'},
                       yaxis={'title': 'Coherence'},
                       margin={'l': 40, 'b': 40, 't': 50, 'r': 50},
                       hovermode='closest',
                       width=150 * graph_width, height=100 * graph_height)

    fig = go.Figure(data=data, layout=layout)

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
              State('start_1', 'value'),
              State('end_1', 'value'),
              State('start_1', 'options'),
              State('loading_data', 'children'))
def update_graph(signal1, schem_filter, schem_sigs, is_merged, k, graph_width, graph_height, mode, eps,
                 t_start, t_end, time, loading_data):
    gmode = 'lines+markers' if mode == 'LM' else 'lines'
    data = []
    if signal1:
        val1 = 0 if (t_start == 0) or (t_start is None) else t_start
        val2 = max(len(time) - 1, 0) if (t_end == 0) or (t_end is None) else t_end
        time1 = time[val1]['label']
        time2 = time[val2]['label']
        dff = loaddata.select_dff_by_time(loading_data, time1, time2, 0)
        cols = dff.columns
        if schem_filter == 'SM' and not (k is None):
            dff = prepare.smoothing_symm(dff, signal1, k, 1)

        if 'SG' in schem_sigs:
            data.append(go.Scatter(x=dff[cols[0]], y=dff[signal1], mode=gmode, name='input'))

        if 'EX' in schem_sigs:
            # all extremes
            dff = schematisation.get_extremes(dff, signal1)
            data.append(go.Scatter(x=dff[cols[0]], y=dff[signal1], mode=gmode, name='extremes'))

        if 'MG' in is_merged and not (eps is None):
            dff = schematisation.get_merged_extremes(dff, signal1, eps)
            if 'MG' in schem_sigs:
                data.append(go.Scatter(x=dff[cols[0]], y=dff[signal1], mode=gmode, name='merge'))

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
              Input('schem_sigs_prepare', 'value'),
              Input('smoothing_window_schem', 'value'),
              Input('graph_width4', 'value'),
              Input('graph_height4', 'value'),
              Input('amplitude_width_input', 'value'),
              Input('class_min_input', 'value'),
              Input('class_max_input', 'value'),
              Input('class_min_input2', 'value'),
              Input('class_max_input2', 'value'),
              Input('class_number', 'value'),
              Input('corr_table_code', 'value'),
              State('start_1', 'value'),
              State('end_1', 'value'),
              State('start_1', 'options'),
              State('loading_data', 'children'))
def update_graph(signal1, schem_filter, is_merged, k, graph_width, graph_height, eps,
                 class_min, class_max, class_min2, class_max2, m, code, t_start, t_end, time, loading_data):
    tbl = [pd.DataFrame()]
    x_title = ''
    y_title = ''
    if signal1:
        val1 = 0 if (t_start == 0) or (t_start is None) else t_start
        val2 = max(len(time) - 1, 0) if (t_end == 0) or (t_end is None) else t_end
        time1 = time[val1]['label']
        time2 = time[val2]['label']
        dff = loaddata.select_dff_by_time(loading_data, time1, time2, 0)
        if schem_filter == 'SM' and not (k is None):
            dff = prepare.smoothing_symm(dff, signal1, k, 1)
        if 'MG' in is_merged and not (eps is None):
            dff = schematisation.get_merged_extremes(dff, signal1, eps)
        else:
            dff = schematisation.get_extremes(dff, signal1)
        # cycles = schematisation.pick_cycles_as_df(sig, signal1)
        cycles_numbers = schematisation.pick_cycles_point_number_as_df(dff, signal1)
        cycles = schematisation.calc_cycles_parameters_by_numbers(dff, signal1, cycles_numbers)
        if m is None:
            pass
        else:
            if code == 'MM':
                tbl = schematisation.correlation_table_with_traces_2(cycles, 'Max', 'Min', mmin_set1=class_min,
                                                                     mmax_set1=class_max, mmin_set2=class_min2,
                                                                     mmax_set2=class_max2, count=m)
                x_title = 'Min'
                y_title = 'Max'
            if code == 'MR':
                tbl = schematisation.correlation_table_with_traces_2(cycles, 'Mean', 'Range', mmin_set1=class_min,
                                                                     mmax_set1=class_max, mmin_set2=class_min2,
                                                                     mmax_set2=class_max2, count=m)
                x_title = 'Range'
                y_title = 'Mean'

    fig = go.Figure()
    # fig = px.imshow(tbls[0], color_continuous_scale='GnBu')
    # fig = make_subplots(cols=1, rows=1, subplot_titles=['Cycles Count'])
    fig.add_trace(go.Heatmap(x=tbl[0].columns, y=tbl[0].index, z=tbl[0].values, colorscale='GnBu'))

    fig.update_layout(width=150 * graph_width, height=100 * graph_height,
                      # margin=dict(l=60, r=60, b=10, t=10),
                      xaxis={'title': x_title}, yaxis={'title': y_title})
    fig.update_xaxes(tickangle=-90)

    return fig


@app.callback(Output('trace_map', 'figure'),
              Input('schematisation', 'value'),
              Input('traces', 'value'),
              Input('schem_filter', 'value'),
              Input('schem_sigs_prepare', 'value'),
              Input('smoothing_window_schem', 'value'),
              Input('graph_width6', 'value'),
              Input('graph_height6', 'value'),
              Input('amplitude_width_input', 'value'),
              Input('dt_max_input', 'value'),
              Input('class_min_input', 'value'),
              Input('class_max_input', 'value'),
              Input('class_min_input2', 'value'),
              Input('class_max_input2', 'value'),
              Input('class_number', 'value'),
              Input('corr_table_code', 'value'),
              State('start_1', 'value'),
              State('end_1', 'value'),
              State('start_1', 'options'),
              State('loading_data', 'children'))
def update_graph(signal1, traces, schem_filter, is_merged, k, graph_width, graph_height, eps, dt_max,
                 class_min, class_max, class_min2, class_max2, m, code, t_start, t_end, time, loading_data):
    fig = go.Figure()
    x_title = ''
    y_title = ''
    if traces is None:
        pass
    elif len(traces) > 2:
        pass
    else:
        tbls = [pd.DataFrame() for i in range(len(traces))]
        fig = make_subplots(rows=1, cols=len(traces), subplot_titles=traces, horizontal_spacing=0.25)
        if signal1:
            val1 = 0 if (t_start == 0) or (t_start is None) else t_start
            val2 = max(len(time) - 1, 0) if (t_end == 0) or (t_end is None) else t_end
            time1 = time[val1]['label']
            time2 = time[val2]['label']
            dff = loaddata.select_dff_by_time(loading_data, time1, time2, 0)
            if schem_filter == 'SM' and not (k is None):
                dff = prepare.smoothing_symm(dff, signal1, k, 1)
            if 'MG' in is_merged and not (eps is None):
                dff = schematisation.get_merged_extremes(dff, signal1, eps)
            else:
                dff = schematisation.get_extremes(dff, signal1)
            cycles_numbers = schematisation.pick_cycles_point_number_as_df(dff, signal1)
            cycles = schematisation.calc_cycles_parameters_by_numbers(dff, signal1, cycles_numbers, traces, dt_max)
            if m is None:
                pass
            else:
                if code == 'MM':
                    tbls = schematisation.correlation_table_with_traces_2(cycles, 'Max', 'Min', traces,
                                                                          mmin_set1=class_min,
                                                                          mmax_set1=class_max, mmin_set2=class_min2,
                                                                          mmax_set2=class_max2, count=m)
                    x_title = 'Min'
                    y_title = 'Max'
                if code == 'MR':
                    tbls = schematisation.correlation_table_with_traces_2(cycles, 'Mean', 'Range', traces,
                                                                          mmin_set1=class_min,
                                                                          mmax_set1=class_max, mmin_set2=class_min2,
                                                                          mmax_set2=class_max2, count=m)
                    x_title = 'Range'
                    y_title = 'Mean'
            x_pos = {1: [1.02], 2: [0.395, 1.02]}
            for i in range(len(traces)):
                ccl = go.Heatmap(x=tbls[1][i].columns, y=tbls[1][i].index, z=tbls[1][i].values,
                                 colorscale='GnBu', colorbar=dict(x=x_pos[len(traces)][i]))
                fig.add_trace(ccl, row=1, col=i + 1)
    if traces is None:
        pass
    else:
        for i in range(len(traces)):
            fig.update_xaxes(title_text=x_title, row=1, col=i + 1)
            fig.update_yaxes(title_text=y_title, row=1, col=i + 1)

    fig.update_layout(width=150 * graph_width, height=100 * graph_height)
    fig.update_xaxes(tickangle=-90)
    return fig


@app.callback(Output('distribution', 'figure'),
              Output('cut1', 'value'),
              Output('cut1_input', 'value'),
              Output('cut2', 'value'),
              Output('cut2_input', 'value'),
              Input('estimation', 'value'),
              Input('graph_width5', 'value'),
              Input('graph_height5', 'value'),
              Input('cut1', 'value'),
              Input('cut1_input', 'value'),
              Input('cut2', 'value'),
              Input('cut2_input', 'value'),
              Input('loading_corr', 'children'),
              Input('distribution', 'clickData'),
              State('corr_code', 'value'))
def update_graph(key, graph_width, graph_height, cut1, cut1_input, cut2, cut2_input, loading_data, click_data, code):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    new_cut1 = cut1
    new_input1 = cut1_input
    new_cut2 = cut2
    new_input2 = cut2_input

    if trigger_id == 'cut1':
        new_input1 = cut1
    if trigger_id == 'cut1_input':
        new_cut1 = cut1_input
    if trigger_id == 'cut2':
        new_input2 = cut2
    if trigger_id == 'cut2_input':
        new_cut2 = cut2_input

    if code == 'MM':
        x_title = 'Min'
        y_title = 'Max'
    else:
        x_title = 'Range'
        y_title = 'Mean'

    fig = go.Figure()
    if key is None:
        pass
    else:
        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{"rowspan": 2}, {}],
                   [None, {}]],
            subplot_titles=("Correlation Table", key, key), horizontal_spacing=0.25)
        data = json.loads(loading_data)
        df = pd.read_json(data[key], orient='split')
        if df.empty:
            pass
        else:
            rows, cols = df.shape
            hist1_data = df.index[cut1 - 1]
            hist2_data = df.columns[cut2 - 1]
            if trigger_id == 'distribution':
                hist1_data = click_data['points'][0]['y']
                hist2_data = click_data['points'][0]['x']
                new_cut1 = df.index.get_loc(hist1_data) + 1
                new_input1 = new_cut1
                new_cut2 = df.columns.get_loc(hist2_data) + 1
                new_input2 = new_cut2

            if rows * cols > 0:
                hist1 = df.loc[hist1_data].to_numpy()
                hist2 = df[hist2_data].to_numpy()

                hist1_range = df.columns.to_numpy()
                hist2_range = df.index.to_numpy()

                hist1_fix = df.index.to_numpy()[new_cut1 - 1]
                hist2_fix = df.columns.to_numpy()[new_cut2 - 1]
                # print('hist1={}'.format(hist1))
                # print('hist2={}'.format(hist2))
                fig.add_trace(go.Heatmap(x=df.columns, y=df.index, z=df.values, colorscale='gnbu',
                                         colorbar=dict(x=0.395)),
                              row=1, col=1)
                fig.add_trace(go.Bar(x=hist1_range, y=hist1, marker_color='rgb(8,64,129)',
                                     name=y_title + '=' + str(hist1_fix)),
                              row=1, col=2)
                fig.add_trace(go.Bar(x=hist2_range, y=hist2, marker_color='rgb(153,215,186)',
                                     name=x_title + '=' + str(hist2_fix)),
                              row=2, col=2)

        fig.update_layout(xaxis={'title': x_title},
                          yaxis={'title': y_title},
                          margin={'l': 40, 'b': 40, 't': 50, 'r': 50},
                          hovermode='closest', clickmode='event',
                          width=150 * graph_width, height=100 * graph_height,
                          plot_bgcolor='rgb(247,252,240)')
        fig.update_xaxes(tickangle=-90)

    return [fig, new_cut1, new_input1, new_cut2, new_input2]


@app.callback(Output('integral', 'figure'),
              Input('cut1', 'value'),
              Input('graph_height7', 'value'),
              Input('graph_width7', 'value'),
              Input('loading_corr', 'children'),
              State('corr_code', 'value'))
def graph_cumulative_distribution(cut1, graph_height, graph_width, loading_data, code):
    fig = go.Figure()
    x_title = 'Range'
    y_title = 'Cumulative distribution, log'
    if code == 'MM':
        pass
    else:
        if loading_data:
            data = json.loads(loading_data)
            if 'cycles' in list(data.keys()):
                df = pd.read_json(data['cycles'], orient='split')
                rows, cols = df.shape
                if rows * cols > 0:
                    hist1_data = df.index[cut1 - 1]
                    hist1 = df.loc[hist1_data].to_numpy()
                    hist1_range = df.columns.to_numpy()
                    dff = schematisation.cumulative_frequency(hist1_range, [hist1], ['cycles'])
                    fig.add_trace(go.Scatter(x=dff['Range'], y=dff['CDF']))
                    fig.update_yaxes(type='log')
    fig.update_layout(xaxis={'title': x_title},
                      yaxis={'title': y_title},
                      margin={'l': 40, 'b': 40, 't': 50, 'r': 50},
                      hovermode='closest', width=150 * graph_width, height=100 * graph_height,
                      plot_bgcolor='rgb(247,252,240)')
    return fig


@app.callback(Output('link-signals', 'href'),
              Output('link-signals', 'hidden'),
              Input('all_signal', 'value'),
              Input('pick_signals', 'n_clicks'),
              Input('signal_1', 'value'),
              Input('signal_filter', 'value'),
              Input('smoothing_window', 'value'),
              Input('start_1', 'value'),
              Input('end_1', 'value'),
              State('loading_data', 'children'),
              State('start_1', 'options'))
def export_signal(all_check, n_clicks, yy, signal_filter, smoothing,
                  t_start, t_end, loading_data, time):
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    # button click
    if triggered_id == 'pick_signals':
        val1 = 0 if (t_start == 0) or (t_start is None) else t_start
        val2 = max(len(time) - 1, 0) if (t_end == 0) or (t_end is None) else t_end
        time1 = time[val1]['label']
        time2 = time[val2]['label']
        dff = loaddata.select_dff_by_time(loading_data, time1, time2, 0)
        export_cols = dff.columns
        if all_check is None or all_check == []:
            dff = dff[[dff.columns[0]] + yy]
            export_cols = yy
        # request smoothing signals
        if 'SM' in signal_filter:
            dff = prepare.set_smoothing_symm(dff, export_cols, smoothing, 1)
        if 'HW' in signal_filter:
            dff = prepare.set_correction_hann(dff, export_cols)

        csv_string = dff.to_csv(index=False, encoding='utf-8')
        csv_string = "data:text/csv;charset=utf-8,%EF%BB%BF" + urllib.parse.quote(csv_string)
        return [csv_string, False]
    # change
    return ["data:text/csv;charset=utf-8,%EF%BB%BF", True]


@app.callback(Output('link-spectrum', 'href'),
              Output('link-spectrum', 'hidden'),
              Input('all_spectrum', 'value'),
              Input('pick_spectrum', 'n_clicks'),
              Input('spectrum_1', 'value'),
              Input('spectrum_2', 'value'),
              Input('start_1', 'value'),
              Input('end_1', 'value'),
              Input('spectrum_filter', 'value'),
              Input('smoothing_window_spectrum', 'value'),
              State('loading_data', 'children'),
              State('start_1', 'options'))
def export_cross_spectrum(all_checked, n_clicks, spectrum_1, spectrum_2, t_start, t_end, spectrum_filter, k,
                          loading_data, time):
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    # button click
    if triggered_id == 'pick_spectrum':
        val1 = 0 if (t_start == 0) or (t_start is None) else t_start
        val2 = max(len(time) - 1, 0) if (t_end == 0) or (t_end is None) else t_end
        time1 = time[val1]['label']
        time2 = time[val2]['label']

        smoothing = k if 'SM' in spectrum_filter else -1
        win = "hann" if 'HW' in spectrum_filter else "boxcar"
        dfres = pd.DataFrame()
        if not (loading_data is None):
            dff = loaddata.select_dff_by_time(loading_data, time1, time2, 0)
            if all_checked is None or all_checked == []:
                dfres = pipeline.calc_signals_cross_spectrum(dff, spectrum_1, spectrum_2, smoothing, win)
            else:
                dfres = pipeline.calc_set_of_signals_cross_spectrum(dff, smoothing, win)
        csv_string = dfres.to_csv(index=False, encoding='utf-8')
        csv_string = "data:text/csv;charset=utf-8,%EF%BB%BF" + urllib.parse.quote(csv_string)
        return [csv_string, False]
    return ["data:text/csv;charset=utf-8,%EF%BB%BF", True]


@app.callback(Output('link-coherence', 'href'),
              Output('link-coherence', 'hidden'),
              Input('all_coherence', 'value'),
              Input('pick_coherence', 'n_clicks'),
              Input('coherence_1', 'value'),
              Input('coherence_2', 'value'),
              Input('start_1', 'value'),
              Input('end_1', 'value'),
              Input('coherence_filter', 'value'),
              Input('smoothing_window_coherence', 'value'),
              Input('segment_len', 'value'),
              State('loading_data', 'children'),
              State('start_1', 'options'))
def export_coherence(all_checked, n_clicks, coherence_1, coherence_2, t_start, t_end, spectrum_filter,
                     k, npseg, loading_data, time):
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    # button click
    if triggered_id == 'pick_coherence':
        dfres = pd.DataFrame()
        # set time range if None
        val1 = 0 if (t_start == 0) or (t_start is None) else t_start
        val2 = max(len(time) - 1, 0) if (t_end == 0) or (t_end is None) else t_end
        time1 = time[val1]['label']
        time2 = time[val2]['label']
        smoothing = k if 'SM' in spectrum_filter else -1
        win = "hann" if 'HW' in spectrum_filter else "boxcar"
        if loading_data:
            df = pd.read_json(loading_data, orient='split')
            if not df.empty:
                dff = loaddata.select_dff_by_time(loading_data, time1, time2, 0)
                if all_checked is None or all_checked == []:
                    dfres = pipeline.calc_signals_coherence(dff, coherence_1, coherence_2, smoothing, win, npseg)
                else:
                    dfres = pipeline.calc_set_of_signals_coherence(dff, smoothing, win, npseg)
        csv_string = dfres.to_csv(index=False, encoding='utf-8')
        csv_string = "data:text/csv;charset=utf-8,%EF%BB%BF" + urllib.parse.quote(csv_string)
        return [csv_string, False]
    return ["data:text/csv;charset=utf-8,%EF%BB%BF", True]


@app.callback(Output('link-cycles', 'href'),
              Output('link-cycles', 'hidden'),
              Input('pick_cycles', 'n_clicks'),
              Input('schematisation', 'value'),
              Input('schem_filter', 'value'),
              Input('schem_sigs_prepare', 'value'),
              Input('smoothing_window_schem', 'value'),
              Input('amplitude_width_input', 'value'),
              State('start_1', 'value'),
              State('end_1', 'value'),
              State('start_1', 'options'),
              State('loading_data', 'children'))
def export_cycles(n_clicks, signal1, schem_filter, is_merged, k, eps, t_start, t_end, time, loading_data):
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    # button click
    if triggered_id == 'pick_cycles':
        sig = pd.DataFrame()
        val1 = 0 if (t_start == 0) or (t_start is None) else t_start
        val2 = max(len(time) - 1, 0) if (t_end == 0) or (t_end is None) else t_end
        time1 = time[val1]['label']
        time2 = time[val2]['label']
        if signal1:
            dff = loaddata.select_dff_by_time(loading_data, time1, time2, 0)
            cols = dff.columns
            sig = dff[[cols[0], signal1]]
            if schem_filter == 'SM':
                sig = prepare.smoothing_symm(sig, signal1, k, 1)

            if 'MG' in is_merged and not (eps is None):
                sig = schematisation.get_merged_extremes(sig, signal1, eps)
            else:
                sig = schematisation.get_extremes(sig, signal1)
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
              Input('traces', 'value'),
              Input('schem_filter', 'value'),
              Input('schem_sigs_prepare', 'value'),
              Input('smoothing_window_schem', 'value'),
              Input('amplitude_width_input', 'value'),
              Input('dt_max_input', 'value'),
              Input('class_min_input', 'value'),
              Input('class_max_input', 'value'),
              Input('class_min_input2', 'value'),
              Input('class_max_input2', 'value'),
              Input('class_number', 'value'),
              Input('corr_table_code', 'value'),
              State('start_1', 'value'),
              State('end_1', 'value'),
              State('start_1', 'options'),
              State('loading_data', 'children'))
def export_table(n_clicks, signal1, traces, schem_filter, is_merged, k, eps, dt_max, class_min, class_max,
                 class_min2, class_max2, m, code,
                 t_start, t_end, time, loading_data):
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    # button click
    if triggered_id == 'pick_table':
        dff = pd.DataFrame()
        val1 = 0 if (t_start == 0) or (t_start is None) else t_start
        val2 = max(len(time) - 1, 0) if (t_end == 0) or (t_end is None) else t_end
        time1 = time[val1]['label']
        time2 = time[val2]['label']
        if signal1:
            dff = loaddata.select_dff_by_time(loading_data, time1, time2, 0)

            if schem_filter == 'SM':
                dff = prepare.smoothing_symm(dff, signal1, k, 1)
            if 'MG' in is_merged and not (eps is None):
                dff = schematisation.get_merged_extremes(dff, signal1, eps)
            else:
                dff = schematisation.get_extremes(dff, signal1)
            cycles_numbers = schematisation.pick_cycles_point_number_as_df(dff, signal1)
            if traces is None:
                traces = []
            cycles = schematisation.calc_cycles_parameters_by_numbers(dff, signal1, cycles_numbers, traces, dt_max)
            if m is None:
                pass
            else:
                if code == 'MM':
                    tbls = schematisation.correlation_table_with_traces_2(cycles, 'Max', 'Min', traces,
                                                                          mmin_set1=class_min,
                                                                          mmax_set1=class_max, mmin_set2=class_min2,
                                                                          mmax_set2=class_max2, count=m)

                if code == 'MR':
                    tbls = schematisation.correlation_table_with_traces_2(cycles, 'Mean', 'Range', traces,
                                                                          mmin_set1=class_min,
                                                                          mmax_set1=class_max, mmin_set2=class_min2,
                                                                          mmax_set2=class_max2, count=m)
                # collect all tables to list
                data = [tbls[0].to_json(date_format='iso', orient='split')]
                for i in range(len(traces)):
                    data.append(tbls[1][i].to_json(date_format='iso', orient='split'))
                dff = pd.DataFrame(data, columns=[code], index=['cycles'] + traces)
        csv_string = dff.to_csv(index=True, encoding='utf-8')
        csv_string = "data:text/csv;charset=utf-8,%EF%BB%BF" + urllib.parse.quote(csv_string)
        return [csv_string, False]
    # change something
    return ["data:text/csv;charset=utf-8,%EF%BB%BF", True]


@app.callback(Output('link-cdf', 'href'),
              Output('link-cdf', 'hidden'),
              Input('cdf', 'n_clicks'),
              Input('cut1', 'value'),
              State('loading_corr', 'children'),
              State('corr_code', 'value'))
def export_table(n_clicks, cut1, loading_data, code):
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    # button click
    if triggered_id == 'cdf':
        if code == 'MR':
            if loading_data:
                data = json.loads(loading_data)
                h = []
                h_r = []
                for key in data.keys():
                    df = pd.read_json(data[key], orient='split')
                    h_data = df.index[cut1 - 1]
                    h.append(df.loc[h_data].to_numpy())
                    h_r.append(df.columns.to_numpy())

                dff = schematisation.cumulative_frequency(h_r[0], h, list(data.keys()), True)
                csv_string = dff.to_csv(index=False, encoding='utf-8')
                csv_string = "data:text/csv;charset=utf-8,%EF%BB%BF" + urllib.parse.quote(csv_string)
                return [csv_string, False]
    # change something
    return ["data:text/csv;charset=utf-8,%EF%BB%BF", True]



@app.callback(Output('graph_width6', 'value'),
              Output('graph_height6', 'value'),
              Input('traces', 'value'),
              State('graph_width6', 'value'),
              State('graph_height6', 'value'))
def set_graph_size(traces, w, h):
    if traces is None:
        return [w, h]
    elif len(traces) == 1:
        return [4, 6]
    else:
        return [9, 6]


@app.callback(Output('input_stats', 'children'),
              Input('schematisation', 'value'),
              State('start_1', 'value'),
              State('end_1', 'value'),
              State('start_1', 'options'),
              State('loading_data', 'children'))
def print_input_stats(signal1, t_start, t_end, time, loading_data):
    inp_str = ''

    if signal1:
        val1 = 0 if (t_start == 0) or (t_start is None) else t_start
        val2 = max(len(time) - 1, 0) if (t_end == 0) or (t_end is None) else t_end
        time1 = time[val1]['label']
        time2 = time[val2]['label']
        dff = loaddata.select_dff_by_time(loading_data, time1, time2, 0)
        s_mean, s_variance, s_deviation, s_koef = schematisation.input_stats(dff, signal1)
        inp_str = 'Input statistics: Mean value is {:.3e}, standard deviation is {:.3e}, ' \
                  'irregular coefficient is {:.3e}'.format(s_mean, s_deviation, s_koef)
    return inp_str


@app.callback(Output('f_max', 'children'),
              Input('schematisation', 'value'),
              Input('schem_sigs_prepare', 'value'),
              Input('amplitude_width_input', 'value'),
              Input('frequency_est', 'value'),
              State('start_1', 'value'),
              State('end_1', 'value'),
              State('start_1', 'options'),
              State('loading_data', 'children'))
def print_max_frequency_estimation(signal1, is_merged, eps, n, t_start, t_end, time, loading_data):
    inp_str = ''

    if signal1:
        val1 = 0 if (t_start == 0) or (t_start is None) else t_start
        val2 = max(len(time) - 1, 0) if (t_end == 0) or (t_end is None) else t_end
        time1 = time[val1]['label']
        time2 = time[val2]['label']
        dff = loaddata.select_dff_by_time(loading_data, time1, time2, 0)
        cols = dff.columns
        sig = dff[[cols[0], signal1]]

        if 'MG' in is_merged:
            f = schematisation.max_frequency(sig, signal1, n, eps)
        else:
            f = schematisation.max_frequency(sig, signal1, n)
        if type(f) is float:
            inp_str = 'Max frequency is approximately {:.3e}'.format(f)
        else:
            inp_str = f
    return inp_str


@app.callback(Output('amplitude_width', 'value'),
              Output('amplitude_width_input', 'value'),
              Output('amplitude_width_input', 'max'),
              Output('schem_sigs_prepare', 'value'),
              Input('amplitude_width', 'value'),
              Input('amplitude_width_input', 'value'),
              Input('schematisation', 'value'),
              State('start_1', 'value'),
              State('end_1', 'value'),
              State('start_1', 'options'),
              State('amplitude_width_input', 'max'),
              State('loading_data', 'children'),
              State('schem_sigs', 'value'))
def amplitude_filter(sldr, inpt, signal1, t_start, t_end, time, cur_max, loading_data, current_sigs):

    # update amplitude filter and input value if signal changed
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    current_slider = sldr
    current_inpt = inpt
    current_range = cur_max

    if trigger_id == 'schematisation':
        if signal1 and sldr:
            val1 = 0 if (t_start == 0) or (t_start is None) else t_start
            val2 = max(len(time) - 1, 0) if (t_end == 0) or (t_end is None) else t_end
            time1 = time[val1]['label']
            time2 = time[val2]['label']
            dff = loaddata.select_dff_by_time(loading_data, time1, time2, 0)
            current_range = dff[signal1].max() - dff[signal1].min()
            current_inpt = current_slider * current_range / 100.0

    if trigger_id == 'amplitude_width' and sldr:
        current_sigs += ['MG']
        if signal1:
            current_inpt = current_slider * current_range / 100.0

    if trigger_id == 'amplitude_width_input' and inpt:
        current_sigs += ['MG']
        if signal1:
            current_slider = current_inpt / current_range * 100

    # print('sldr={}, inpt={}, range={}'.format(current_slider, current_inpt, current_range))

    return [current_slider, current_inpt, current_range, current_sigs]


@app.callback(Output('loading_corr', 'children'),
              Output('filenames', 'children'),
              Output('cut1', 'max'),
              Output('cut1_input', 'max'),
              Output('cut2', 'max'),
              Output('cut2_input', 'max'),
              Output('estimation', 'options'),
              Output('corr_code', 'value'),
              Input('upload_tables', 'contents'),
              State('upload_tables', 'filename'))
def upload_file(contents, filenames):
    data = {}
    nm = ''
    classes = 3
    options = []
    code = 'MM'
    if contents:
        data, code, opts, classes = loaddata.load_and_ave_set(contents, filenames)
        nm = loaddata.load_files(filenames)
        options = get_options(opts)
    return [json.dumps(data, indent=0),
            nm, classes, classes, classes, classes, options, code]


@app.callback(Output('range1', 'children'),
              Output('range2', 'children'),
              Input('corr_table_code', 'value'))
def update_labels(code):
    if code == 'MM':
        return ['max', 'min']
    else:
        return ['mean', 'range']


@app.callback(Output('schem_sigs', 'value'),
              Input('schem_sigs_prepare', 'value'),
              State('schem_sigs', 'value'))
def not_draw_merge(is_merged, sigs):
    current_sigs = sigs
    if 'MG' in is_merged:
        current_sigs.append('MG')
    else:
        current_sigs = [x for x in sigs if x != 'MG']
    return current_sigs


if __name__ == '__main__':
    app.run_server(debug=True)
