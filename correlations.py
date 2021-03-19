from staff import pipeline
import os
import sys

"""
"""





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

    delimiter = ','
    df = pipeline.read_data(filename, sigs + add_traces, delimiter)
    if type(df) == str:
        print(df)
        exit()

    print("Load data Ok")

    print(pipeline.check_folders_tree(mode, sigs))
    os.chdir(os.getcwd() + '/' + mode)

    # prepare parameters for processing
    # smoothing window sizes, length of array = number of signals
    k = [2] * len(sigs)

    # hann-weighting, length of array = number of signals
    hann = [True] * len(sigs)

    # amplitude filter values, length of array = number of signals
    eps = [0.5] * len(sigs)

    # array of additional tracing parameters, length of array = number of signals
    # should be same for the picked mode for picked signal for all flights
    traces = [['vel', 'tgag']] * len(sigs)

    # array of time for averaging additional tracing parameters, length of array = number of signals
    # None if unconditional averaging
    dt_max = [0.1] * len(sigs)

    # array of global minimum for Mean & Range,  length of array = number of signals
    class_min1 = [-5.0] * len(sigs)
    class_min2 = [-5.0] * len(sigs)
    # array of global maximum for Mean & Range, length of array = number of signals
    class_max1 = [5.0] * len(sigs)
    class_max2 = [5.0] * len(sigs)

    # array of classes count, length of array = number of signals
    m = [10] * len(sigs)

    status = pipeline.processing_parameters_set(flight, df, sigs, k, hann, eps, traces, dt_max, class_min1, class_max1,
                                                class_min2, class_max2, m)
    print(status)
