import pandas as pd
import base64
import io


def parse_data(contents, filename, index=None):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    df = pd.DataFrame()
    if 'txt' in filename:
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')), index_col=index)
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

    df = parse_data(contents[0], names[0], index=0)
    code = df.columns.to_numpy()[0]
    options = df.index.tolist()
    # print('options={}'.format(options))
    data = {}
    classes = 1.0
    for opt in options:
        dfi = pd.read_json(df.loc[opt].values[0], orient='split')
        classes, _ = dfi.shape
        data[opt] = dfi

    for i in range(1, len(names)):
        df = parse_data(contents[i], names[i], index=0)
        for opt in options:
            dfi = pd.read_json(df.loc[opt].values[0], orient='split')
            data[opt] += dfi

    for opt in options:
        data[opt] /= len(names)

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
