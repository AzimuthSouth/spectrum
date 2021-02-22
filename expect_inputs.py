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
from staff import pipeline
from scipy import signal
import urllib

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
            html.Label("Smoothing window size"),
            html.Div(dcc.Slider(
                id='smoothing_window',
                min=1,
                max=100,
                value=3,
                marks={str(i): str(i) for i in range(1, 101)},
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
            html.Label("Smoothing window size"),
            html.Div(dcc.Slider(
                id='smoothing_window_spectrum',
                min=1,
                max=10,
                value=3,
                marks={str(i): str(i) for i in range(0, 10)},
                step=None), style={'width': '50%', 'padding': '0px 20px 20px 20px'}),

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
        cols = df.columns
        trp = prepare.calc_time_range(df[cols[0]].to_numpy())
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
              State('t_start', 'max'),
              State('t_start', 'step'))
def update_graph(signal_1, signal_filter, k, graph_width, graph_height,
                 t_start, t_end, t_range, loading_data, t_min, t_max, t_step):
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

    data = []
    if signal_1:
        df = pd.read_json(loading_data, orient='split')
        cols = df.columns
        dff = df[(df[cols[0]] >= (val1 - dt)) & (df[cols[0]] <= (val2 + dt))]
        dff.reset_index(drop=True, inplace=True)
        for yy in signal_1:
            sig = dff[[cols[0], yy]]
            print(sig)

            data.append(go.Scatter(x=sig[cols[0]], y=sig[yy], mode='lines+markers', name=yy))

            if 'SM' in signal_filter:
                sig = prepare.smoothing_symm(sig, yy, k, 1)
                data.append(go.Scatter(x=sig[cols[0]], y=sig[yy], mode='lines+markers', name='smooth'))

            if 'HW' in signal_filter:
                sig = prepare.correction_hann(sig, yy)
                data.append(go.Scatter(x=sig[cols[0]], y=sig[yy], mode='lines+markers', name='hann_correction'))

    layout = go.Layout(xaxis={'title': 'Time'},
                       yaxis={'title': 'Input'},
                       margin={'l': 40, 'b': 40, 't': 50, 'r': 50},
                       hovermode='closest',
                       width=150 * graph_width, height=100 * graph_height)

    fig = go.Figure(data=data, layout=layout)

    return [fig, [val1, val2], val1, val2]


@app.callback(Output('link-signals', 'href'),
              Output('link-signals', 'hidden'),
              Input('pick_signals', 'n_clicks'),
              Input('t_start', 'value'),
              Input('t_end', 'value'),
              Input('time_range_slider', 'value'),
              State('loading_data', 'children'),
              State('t_start', 'min'),
              State('t_start', 'max'),
              State('t_start', 'step'))
def export_signal(n_clicks, t_start, t_end, t_range, loading_data, t_min, t_max, t_step):
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    # button click
    if triggered_id == 'pick_signals':
        print("in export signal")
        print(n_clicks)
        print(loading_data is None)
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
        csv_string = dff.to_csv(index=False, encoding='utf-8')
        csv_string = "data:text/csv;charset=utf-8,%EF%BB%BF" + urllib.parse.quote(csv_string)
        print("string={}".format(csv_string))
        return [csv_string, False]
    # change time
    if triggered_id == 't_start' or triggered_id == 't_end' or triggered_id == 'time_range_slider':
        return ["data:text/csv;charset=utf-8,%EF%BB%BF", True]


@app.callback(Output('spectrum_graph', 'figure'),
              # Output('spectrum_graph1', 'figure'),
              Input('spectrum_1', 'value'),
              Input('spectrum_2', 'value'),
              Input('spectrum_filter', 'value'),
              Input('smoothing_window_spectrum', 'value'),
              Input('graph_width1', 'value'),
              Input('graph_height1', 'value'),
              State('t_start', 'value'),
              State('t_end', 'value'),
              State('t_start', 'step'),
              State('loading_data', 'children'))
def update_graph(spectrum_1, spectrum_2, spectrum_filter, k, graph_width, graph_height,
                 t_start, t_end, t_step, loading_data):
    fig = make_subplots(rows=2, cols=1)
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

        fig.add_trace(go.Scatter(x=f, y=mod, mode='lines+markers', name='cross_spectrum'), row=1, col=1)
        fig.add_trace(go.Scatter(x=f, y=phase, mode='lines+markers', name='phase'), row=2, col=1)
    fig.update_xaxes(title_text="Frequencies", row=1, col=1)
    fig.update_xaxes(title_text="Frequencies", row=2, col=1)
    fig.update_yaxes(title_text="Cross Spectrum Module", row=1, col=1)
    fig.update_yaxes(title_text="Cross Spectrum Phase", row=2, col=1)
    fig.update_layout(width=150 * graph_width, height=100 * graph_height)
    return fig


@app.callback(Output('link-spectrum', 'href'),
              Output('link-spectrum', 'hidden'),
              Input('pick_spectrum', 'n_clicks'),
              Input('t_start', 'value'),
              Input('t_end', 'value'),
              Input('spectrum_filter', 'value'),
              Input('smoothing_window_spectrum', 'value'),
              State('loading_data', 'children'),
              State('t_start', 'min'),
              State('t_start', 'max'),
              State('t_start', 'step'))
def export_cross_spectrum(n_clicks, t_start, t_end, spectrum_filter, k, loading_data, t_min, t_max, t_step):
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
              State('t_start', 'value'),
              State('t_end', 'value'),
              State('t_start', 'step'),
              State('loading_data', 'children'))
def update_graph(coherence_1, coherence_2, coherence_filter, k, segment_len, graph_width, graph_height,
                 t_start, t_end, t_step, loading_data):
    data = []
    if coherence_1 and coherence_2:
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

        data.append(go.Scatter(x=f, y=c_xx, mode='lines+markers', name='cross_spectrum'))

    layout = go.Layout(xaxis={'title': 'Frequencies'},
                       yaxis={'title': 'Coherence'},
                       margin={'l': 40, 'b': 40, 't': 50, 'r': 50},
                       hovermode='closest',
                       width=150 * graph_width, height=100 * graph_height)

    fig = go.Figure(data=data, layout=layout)

    return fig


@app.callback(Output('link-coherence', 'href'),
              Output('link-coherence', 'hidden'),
              Input('pick_coherence', 'n_clicks'),
              Input('t_start', 'value'),
              Input('t_end', 'value'),
              Input('coherence_filter', 'value'),
              Input('smoothing_window_coherence', 'value'),
              Input('segment_len', 'value'),
              State('loading_data', 'children'),
              State('t_start', 'min'),
              State('t_start', 'max'),
              State('t_start', 'step'))
def export_coherence(n_clicks, t_start, t_end, spectrum_filter, k, npseg, loading_data, t_min, t_max, t_step):
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    # button click
    if triggered_id == 'pick_coherence':
        dfres = pd.DataFrame()
        # set time range if None
        val1 = t_min if t_start is None else t_start
        val2 = t_max if t_end is None else t_end
        dt = 0.0 if t_step is None else t_step / 2
        smoothing = k if 'SM' in spectrum_filter else -1
        win = "hann" if 'HW' in spectrum_filter else "boxcar"
        print("in export coh")
        if loading_data:
            df = pd.read_json(loading_data, orient='split')
            if not df.empty:
                cols = df.columns
                dff = df[(df[cols[0]] >= (val1 - dt)) & (df[cols[0]] <= (val2 + dt))]
                dff.reset_index(drop=True, inplace=True)
                dfres = pipeline.calc_set_of_signals_coherence(dff, smoothing, win, npseg)
        csv_string = dfres.to_csv(index=False, encoding='utf-8')
        csv_string = "data:text/csv;charset=utf-8,%EF%BB%BF" + urllib.parse.quote(csv_string)
        return [csv_string, False]
    return ["data:text/csv;charset=utf-8,%EF%BB%BF", True]


if __name__ == '__main__':
    app.run_server(debug=True)
