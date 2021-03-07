import pandas as pd


def load_and_ave_set(names):
    if names is None:
        return pd.DataFrame()
    if len(names) == 0:
        return pd.DataFrame()
    df = pd.read_csv(names[0], index_col=0)
    for name in names[1:]:
        df += pd.read_csv(name, index_col=0)
    df /= len(names)
    return df


def load_files(names):
    s = ''
    for name in names[:-1]:
        s += name + ", "
    s += names[-1]
    return s