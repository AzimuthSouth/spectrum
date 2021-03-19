import pandas as pd
from staff import prepare
from staff import analyse
from staff import schematisation
from scipy import signal
import scipy
import os
from pathlib import Path


def calc_set_of_signals_cross_spectrum(df, smoothing, win):
    """
    Calculate psd and cross spectrum for input signals
    :param df: input dataframe
    :param smoothing: integer, 1-10, smoothing window width$ if -1 - no smoothing
    :param win: text, window function ("hann" or "boxcar")
    :return: df of cross spectrum
    """
    col_names = df.columns
    rows, cols = df.shape
    _, _, dt, _ = prepare.calc_time_range(df[col_names[0]].to_numpy())
    col_names = df.columns
    dff = pd.DataFrame()
    if smoothing > 0:
        rows -= smoothing
    f = scipy.fft.rfftfreq(rows, dt)
    dff["Frequencies"] = f
    for column in col_names[1:]:
        sig = df.copy()
        if smoothing > 0:
            sig = prepare.smoothing_symm(sig, column, smoothing, 1)
        _, g_xy = signal.csd(sig[column], sig[column], (1.0 / dt), window=win, nperseg=rows)
        name = "psd_" + column
        dff[name] = g_xy
    for i in range(1, len(col_names)):
        for j in range(i + 1, len(col_names)):
            sig = df.copy()
            if smoothing > 0:
                sig = prepare.set_smoothing_symm(sig, [col_names[i], col_names[j]], smoothing, 1)
            _, g_xy = signal.csd(sig[col_names[i]], sig[col_names[j]], (1.0 / dt), window=win, nperseg=rows)
            name = col_names[i] + "_" + col_names[j]
            mod, phase = analyse.cross_spectrum_mod_fas(g_xy)
            dff[name + "_module"] = mod
            dff[name + "_phase"] = phase
    return dff


def calc_signals_cross_spectrum(df, name1, name2, smoothing, win):
    """
    Calculate psd and cross spectrum for input signals
    :param df: input dataframe
    :param name1: 1st signal
    :param name2: 2nd signal
    :param smoothing: integer, 1-10, smoothing window width$ if -1 - no smoothing
    :param win: text, window function ("hann" or "boxcar")
    :return: df of cross spectrum
    """
    col_names = df.columns
    rows, cols = df.shape
    _, _, dt, _ = prepare.calc_time_range(df[col_names[0]].to_numpy())
    dff = pd.DataFrame()
    if smoothing > 0:
        rows -= smoothing
    f = scipy.fft.rfftfreq(rows, dt)
    dff["Frequencies"] = f
    sig = df.copy()
    if name1 == name2:
        if smoothing > 0:
            sig = prepare.smoothing_symm(sig, name1, smoothing, 1)
        _, g_xy = signal.csd(sig[name1], sig[name1], (1.0 / dt), window=win, nperseg=rows)
        name = "psd_" + name1
        dff[name] = g_xy
    else:
        if smoothing > 0:
            sig = prepare.set_smoothing_symm(sig, [name1, name2], smoothing, 1)
        _, g_xy = signal.csd(sig[name1], sig[name2], (1.0 / dt), window=win, nperseg=rows)
        name = name1 + "_" + name2
        mod, phase = analyse.cross_spectrum_mod_fas(g_xy)
        dff[name + "_module"] = mod
        dff[name + "_phase"] = phase
    return dff


def calc_set_of_signals_coherence(df, smoothing, win, npseg):
    """
    Calculate psd and cross spectrum for input signals
    :param df: input dataframe
    :param smoothing: integer, 1-10, smoothing window width$ if -1 - no smoothing
    :param win: text, window function ("hann" or "boxcar")
    :param npseg: integer, nperseg for spectrum analysis
    :return: df of cross spectrum
    """
    col_names = df.columns
    rows, cols = df.shape
    _, _, dt, _ = prepare.calc_time_range(df[col_names[0]].to_numpy())
    col_names = df.columns
    dff = pd.DataFrame()
    if smoothing > 0:
        rows -= smoothing
    if npseg > rows:
        npseg = rows
    f = scipy.fft.rfftfreq(npseg, dt)
    dff["Frequencies"] = f
    for i in range(1, len(col_names)):
        for j in range(i + 1, len(col_names)):
            sig = df.copy()
            if smoothing > 0:
                sig = prepare.set_smoothing_symm(df, [col_names[i], col_names[j]], smoothing, 1)
            _, c_xy = signal.coherence(sig[col_names[i]], sig[col_names[j]],
                                       (1.0 / dt), window=win, nperseg=npseg)
            name = col_names[i] + "_" + col_names[j]
            dff[name] = c_xy
    return dff


def calc_signals_coherence(df, name1, name2, smoothing, win, npseg):
    """
    Calculate psd and cross spectrum for input signals
    :param df: input dataframe
    :param name1: 1st signal
    :param name2: 2nd signal
    :param smoothing: integer, 1-10, smoothing window width$ if -1 - no smoothing
    :param win: text, window function ("hann" or "boxcar")
    :param npseg: integer, nperseg for spectrum analysis
    :return: df of cross spectrum
    """
    col_names = df.columns
    rows, cols = df.shape
    _, _, dt, _ = prepare.calc_time_range(df[col_names[0]].to_numpy())
    dff = pd.DataFrame()
    if smoothing > 0:
        rows -= smoothing
    if npseg > rows:
        npseg = rows
    f = scipy.fft.rfftfreq(npseg, dt)
    dff["Frequencies"] = f
    sig = df.copy()
    if name1 == name2:
        if smoothing > 0:
            sig = prepare.smoothing_symm(df, name1, smoothing, 1)
        _, c_xy = signal.coherence(sig[name1], sig[name1], (1.0 / dt), window=win, nperseg=npseg)
    else:
        if smoothing > 0:
            sig = prepare.set_smoothing_symm(df, [name1, name2], smoothing, 1)
        _, c_xy = signal.coherence(sig[name1], sig[name2], (1.0 / dt), window=win, nperseg=npseg)
    name = name1 + "_" + name2
    dff[name] = c_xy
    return dff


def calc_cycles(df, name, eps):
    """

    :param df: input signals dataFrame
    :param name: expected signal column name
    :param eps: class width
    :return: dataFrame with cycles
    """
    # merge signal and pick extremes
    sig = schematisation.merge(df, name, eps)
    sig = schematisation.pick_extremes(sig, name)
    cycles = schematisation.pick_cycles_as_df(sig, name)
    return cycles


def calc_correlation_table(df, name, eps, code, m):
    cycles = calc_cycles(df, name, eps)
    tbl = pd.DataFrame()

    if code == 1:
        tbl = schematisation.correlation_table(cycles, 'Max', 'Min', m)
    if code == 2:
        tbl = schematisation.correlation_table(cycles, 'Range', 'Mean', m)

    return tbl


def read_data(filename, all_signals):
    """
    :param filename: processing file
    :param all_signals: load and traces signals
    :return: dataFrame of processing signals
    """
    curr_dir = os.getcwd()
    df = pd.read_csv(curr_dir + '/' + filename)
    cols = [df.columns[0]] + all_signals
    all_exists = set([sig in df.columns for sig in all_signals])
    if len(all_exists) > 1:
        return "Error! Some signals or traces don't exists in processing file!"
    dff = df[cols]
    return dff


def processing_parameter(df, load_signal, k, hann, eps, traces, dt_max, class_min1, class_max1,
                         class_min2, class_max2, m):
    """
    Processing single parameter
    :param df: dataFrame
    :param load_signal: processing parameter
    :param k: smoothing window
    :param hann: hann-weighting label
    :param eps: amplitude filter
    :param traces: additional parameters for tracing
    :param dt_max: time interval for averaging tracing parameters
    :param class_min1: global minimum for Mean
    :param class_max1: global maximum for Mean
    :param class_min2: global minimum for Range
    :param class_max2: global maximum for Range
    :param m: classes count
    :return: dataFrame with corrtables
    """
    if k > 0:
        df = prepare.set_smoothing_symm(df, [load_signal], k, 1)
    if hann:
        df = prepare.set_correction_hann(df, [load_signal])
    ext = schematisation.get_merged_extremes(df, load_signal, eps)
    cyc_num = schematisation.pick_cycles_point_number_as_df(ext, load_signal)
    cycles = schematisation.calc_cycles_parameters_by_numbers(ext, load_signal, cyc_num, traces, dt_max)
    tbls = schematisation.correlation_table_with_traces_2(cycles, 'Mean', 'Range', traces,
                                                          mmin_set1=class_min1,
                                                          mmax_set1=class_max1, mmin_set2=class_min2,
                                                          mmax_set2=class_max2, count=m)
    data = [tbls[0].to_json(date_format='iso', orient='split')]
    for j in range(len(traces)):
        data.append(tbls[1][j].to_json(date_format='iso', orient='split'))
    dff = pd.DataFrame(data, columns=['MR'], index=['cycles'] + traces)
    return dff


def processing_parameters_set(flight, df, load_signals, k, hann, eps, traces, dt_max, class_min1, class_max1,
                              class_min2, class_max2, m):
    """

    :param flight: flight name
    :param df: dataFrame
    :param load_signals: processing signals
    :param k: array of smoothing windows
    :param hann: hann-weighting label
    :param eps: array of amplitude filter, abs value
    :param traces: array of additional tracing parameters
    :param dt_max: max time interval to averaging tracing parameters
    :param class_min1: Mean global minimums
    :param class_max1: Mean global maximum
    :param class_min2: Range global minimum
    :param class_max2: Range global maximum
    :param m: classes counts
    :return:
    """

    lengths = calc_length([load_signals, k, hann, eps, traces, dt_max, class_min1, class_max1,
                           class_min2, class_max2, m])
    if len(set(lengths)) > 1:
        message = "Error! Counts of processing signals and parameters are different.\n" \
                  "Signals count is {}\n Smoothing windows count is {}\n " \
                  "Hann-weighting labels counts is {}" \
                  "Amplitude filters count is {}\n Traces count is {}\n" \
                  "Max time for averaging count is {}\n Class_min1 count is {}\n" \
                  "Class_max1 count is {}\n Class_min2 count is {}\n Class_max2 count is {}\n" \
                  "Classes counts count is {}\n".format(lengths[0], lengths[1], lengths[2], lengths[3], lengths[4],
                                                        lengths[5], lengths[6], lengths[7], lengths[8], lengths[9],
                                                        lengths[10])
        return message
    try:
        for i in range(lengths[0]):
            print(f"Processing signal: {load_signals[i]}")
            table = processing_parameter(df, load_signals[i], k[i], hann[i], eps[i], traces[i], dt_max[i],
                                         class_min1[i], class_max1[i], class_min2[i], class_max2[i], m[i])
            # export table to folder load_signals[i] (parameter name)
            fname = load_signals[i] + '/' + flight + '.txt'
            table.to_csv(fname, index=True, encoding='utf-8')
            return "Processing Complete"
    except:
        return "Processing Failed"


def calc_length(param):
    return [len(i) for i in param]


def parse_filename(filename):
    parts = filename.split('_')
    flight = parts[0]
    mode = '-'.join(parts[1:]).split('.')[0]
    return [flight, mode]


def get_signals(s):
    s1 = s.replace('/', u"\u005c")
    s1 = [i for i in s1.split(',') if i != '\n']
    return s1


def check_folders_tree(mode, sigs):
    s = ''
    isdir = os.path.exists(os.getcwd() + '/' + mode)
    if not isdir:
        s += f"Create folder {os.getcwd() + '/' + mode}\n"
    for sig in sigs:
        isdir = os.path.exists(os.getcwd() + '/' + mode + '/' + sig)
        if not isdir:
            s += f"Create folder {os.getcwd() + '/' + mode + '/' + sig}\n"

    Path(os.getcwd() + '/' + mode).mkdir(parents=True, exist_ok=True)
    [Path(os.getcwd() + '/' + mode + '/' + sig).mkdir(parents=True, exist_ok=True) for sig in sigs]
    return s + "Folder tree is OK\n"
