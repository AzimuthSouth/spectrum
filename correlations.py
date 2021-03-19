import pandas as pd
from staff import pipeline
from staff import prepare
from research import examples
from staff import schematisation
import numpy
import pandas
import matplotlib.pyplot as plt
import os
import sys

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    os.chdir(os.path.dirname(sys.argv[1]))
    filename = os.path.basename(sys.argv[1])
    print(f"Processing file: {filename}")
    flight, mode = pipeline.parse_filename(filename)
    print(f"Current flight: {flight}, current mode: {mode}")

    # "replace this strings with processing parameters and traces"
    sigs = "input"
    add_traces = "vel,tgag"

    sigs = pipeline.get_signals(sigs)
    add_traces = pipeline.get_signals(add_traces)
    print(f"Processing signals: {sigs}")
    print(f"Tracing signals: {add_traces}")

    df = pipeline.read_data(filename, sigs + add_traces)
    print("Load data Ok")

    print(pipeline.check_folders_tree(mode, sigs))
    os.chdir(os.getcwd() + '/' + mode)
    print(f"Current folder: {os.getcwd()}")

    # prepare parameters for processing
    # smoothing window sizes, length of array = number of signals
    k = [2] * len(sigs)

    # hann-weighting, length of array = number of signals
    hann = [True] * len(sigs)

    # amplitude filter values, length of array = number of signals
    eps = [None] * len(sigs)

    # array of additional tracing parameters, length of array = number of signals
    # should be same for the picked mode for picked signal for all flights
    traces = [['vel', 'tgag']] * len(sigs)

    # array of time for averaging additional tracing parameters, length of array = number of signals
    # None if unconditional averaging
    dt_max = [None] * len(sigs)

    # array of global minimum for Mean & Range,  length of array = number of signals
    class_min1 = [None] * len(sigs)
    class_min2 = [None] * len(sigs)
    # array of global maximum for Mean & Range, length of array = number of signals
    class_max1 = [None] * len(sigs)
    class_max2 = [None] * len(sigs)

    # array of classes count, length of array = number of signals
    m = [10] * len(sigs)

    status = pipeline.processing_parameters_set(flight, df, sigs, k, hann, eps, traces, dt_max, class_min1, class_max1,
                                                class_min2, class_max2, m)
    print(status)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
