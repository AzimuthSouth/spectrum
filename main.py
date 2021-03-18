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
    cols = df.columns

    lenghs = [len(load_signals), len(k), len(eps)]
    if len(lenghs) > 1:
        print("")


    for i in range(len(k)):
        smooth = prepare.set_smoothing_symm(df, cols[1:], k, 1)
        extremes = schematisation.get_merged_extremes(smooth, col, eps)



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    pass

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
