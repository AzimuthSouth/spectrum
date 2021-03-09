import pandas as pd


def load_and_ave_set(names):
    if names is None:
        return [{}, None, [], 1.0]
    if len(names) == 0:
        return [{}, None, [], 1.0]

    # check if all codes are the same
    codes = set([pd.read_csv(name, index_col=0).columns.tolist()[0] for name in names])
    if len(codes) != 1:
        return [{}, None, ["Different Codes!"], 1.0]

    df = pd.read_csv(names[0], index_col=0)
    code = df.columns.to_numpy()[0]
    options = df.index.tolist()
    # print('options={}'.format(options))
    data = {}
    classes = 1.0
    for opt in options:
        dfi = pd.read_json(df.loc[opt].values[0], orient='split')
        classes, _ = dfi.shape
        data[opt] = dfi

    for name in names[1:]:
        df = pd.read_csv(name, index_col=0)
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
