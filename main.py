import pandas as pd
from staff import prepare
from research import examples
from staff import schematisation
import numpy
import pandas
import matplotlib.pyplot as plt
import os


def read_data(filename, all_signals):
    """
    :param filename: processing file
    :param signals: processing signals
    :return: dataFrame of processing signals
    """
    curr_dir = os.getcwd()
    df = pd.read_csv(curr_dir + '/' + filename)
    cols = df.columns[0] + all_signals
    dff = df[cols]
    return dff


def processing(df, load_signals, k, eps):
    """

    :param df: dataFrame
    :param load_signals: processing signals
    :param k: array of smoothing windows
    :param eps: array of amplitude filter, abs value
    :return:
    """
    length = {len(load_signals), len(k), len(eps)}
    if len(length) > 1:
        message = "Error! Numbers of processing signals, smoothing parameters and amplitude filters are different."
        return message

    for i in range(len(k)):
        smooth = prepare.set_smoothing_symm(df, load_signals[i], k[i], 1)
        extremes = schematisation.get_merged_extremes(smooth, load_signals[i], eps[i])
        cycles = schematisation.pick_cycles_as_df(extremes, load_signals[i])




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    pass

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
