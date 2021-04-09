import pandas as pd
import base64
import io
import numpy
import json
from staff import schematisation


def parse_data(contents, filename, index=None):
    df = pd.DataFrame()
    if 'txt' in filename:
        df = universal_upload(contents, index)
    return df


def universal_upload(contents, index=None):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    ln = io.StringIO(decoded.decode('utf-8')).readline()
    df = pd.DataFrame()
    if ',' in ln:
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')), delimiter=",", index_col=index)
    if ';' in ln:
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')), delimiter=";", index_col=index)
        cols = df.columns
        for col in cols:
            coli = df[col].to_numpy()
            sf = [float(s.replace(',', '.')) if type(s) == str else s for s in coli]
            df[col] = sf
    return df


def load_and_ave_set(contents, names):
    if names is None:
        return [{}, None, [], 1.0]
    if len(names) == 0:
        return [{}, None, [], 1.0]

    # check if all codes are the same
    codes = set([parse_data(contents[i], names[i], index=0).columns.tolist()[0] for i in range(len(names))])
    if len(codes) != 1:
        return [{}, None, ["Different Codes!"], 1.0]

    traces = set([len(parse_data(contents[i], names[i], index=0).index) for i in range(len(names))])
    if len(traces) != 1:
        return [{}, None, ["Different Traces!"], 1.0]

    df = parse_data(contents[0], names[0], index=0)
    code = df.columns.to_numpy()[0]
    options = df.index.tolist()
    # print('options={}'.format(options))
    data = {}
    counts_traces = {}
    classes = 1.0
    for opt in options:
        dfi = pd.read_json(df.loc[opt].values[0], orient='split')
        for col in dfi.columns:
            dfi[col] = pd.to_numeric(dfi[col], downcast='float')
        classes, _ = dfi.shape
        counts = numpy.zeros((classes, classes))
        for i in range(classes):
            for j in range(classes):
                if opt == 'cycles':
                    counts[i][j] += 1
                elif dfi.values[i][j] > 0:
                    counts[i][j] += 1
        data[opt] = dfi
        counts_traces[opt] = counts

    for i in range(1, len(names)):
        df = parse_data(contents[i], names[i], index=0)
        for opt in options:
            dfi = pd.read_json(df.loc[opt].values[0], orient='split')
            for col in dfi.columns:
                dfi[col] = pd.to_numeric(dfi[col], downcast='float')
            counts = numpy.zeros((classes, classes))
            for k in range(classes):
                for j in range(classes):
                    if opt == 'cycles':
                        counts[k][j] += 1
                    elif dfi.values[k][j] > 0:
                        counts[k][j] += 1
            data[opt] += dfi
            counts_traces[opt] += counts

    for opt in options:
        for i in range(classes):
            for j in range(classes):
                if counts_traces[opt][i][j] > 0:
                    val = data[opt].values[i][j] / counts_traces[opt][i][j]
                    data[opt]._set_value(data[opt].index[i], data[opt].columns[j], val)

    data_str = {}
    for opt in options:
        data_str[opt] = data[opt].to_json(date_format='iso', orient='split')

    return [data_str, code, options, classes]


def load_files(names):
    s = ''
    for name in names[:-1]:
        s += name + ", "
    s += names[-1]
    return s


def select_dff_by_time(json_data, t_start=None, t_end=None, t_step=None):
    df = pd.read_json(json_data, orient='split')
    cols = df.columns
    val1 = df[cols[0]].iloc[0] if t_start is None else t_start
    val2 = df[cols[0]].iloc[-1] if t_end is None else t_end
    dt = 0.0 if t_step is None else t_step / 2
    dff = df[(df[cols[0]] >= (val1 - dt)) & (df[cols[0]] <= (val2 + dt))]
    dff.reset_index(drop=True, inplace=True)
    return dff


def get_kpi(loading_data, ind):
    data = json.loads(loading_data)
    df = pd.read_json(data['cycles'], orient='split')
    hist1_fix = df.index.to_numpy()[ind]
    h = []
    h_r = []
    for key in data.keys():
        df = pd.read_json(data[key], orient='split')
        h_data = df.index[ind]
        h.append(df.loc[h_data].to_numpy())
        h_r.append(df.columns.to_numpy())

    dff = schematisation.cumulative_frequency(h_r[0], h, list(data.keys()), False)
    csv_string = f"mean={hist1_fix}\n" + dff.to_csv(index=False, encoding='utf-8') + "\n"
    return csv_string


def convert_corr_table_to_excel(df):
    options = df.index.tolist()
    st = ""
    for opt in options:
        st += f"Parameter:{opt}\n"
        dfi = pd.read_json(df.loc[opt].values[0], orient='split')
        st += dfi.to_csv(index=True, encoding='utf-8')
        st += "\n\n"
    return st
