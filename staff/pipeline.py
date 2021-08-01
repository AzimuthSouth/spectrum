import pandas as pd
from staff import prepare
from staff import analyse
from staff import schematisation
from staff import loaddata
from scipy import signal
import scipy
import os
from pathlib import Path
from os import walk
import json
import numpy


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


def read_data(filename, all_signals, delimiter=',', ind=None, code='MR'):
    """
    :param filename: processing file
    :param all_signals: load and traces signals
    :param delimiter: delimiter of the file
    :return: dataFrame of processing signals
    """
    curr_dir = os.getcwd()
    df = pd.read_csv(curr_dir + '/' + filename, delimiter=delimiter, index_col=ind)
    if type(all_signals) == str:
        all_signals = all_signals.split(',')
    else:
        code = df.columns.to_numpy()[0]
        all_signals = []
    cols = [df.columns[0]] + all_signals
    all_exists = set([sig in df.columns.to_numpy() for sig in cols])
    if all_exists != {True}:
        return "Error! Some signals or traces don't exists in processing file!"
    dff = df[cols]
    if delimiter == ';':
        cols = dff.columns
        for col in cols:
            coli = dff[col].to_numpy()
            sf = [float(s.replace(',', '.')) if type(s) == str else s for s in coli]
            dff[col] = sf
    print(dff)
    print(code)
    return [dff, code]


def processing_parameter(df, load_signal, k, hann, eps, traces, dt_max, class_min1, class_max1,
                         class_min2, class_max2, m, code='MR'):
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
    ls = get_signal_back(load_signal)
    l_sig = ls
    if k > 0:
        df = prepare.set_smoothing_symm(df, [ls], k, 1)
        l_sig += '_smooth'

    if hann:
        df = prepare.set_correction_hann(df, [ls])
        l_sig += '_hann'

    ext = schematisation.get_merged_extremes(df, l_sig, eps)
    cyc_num = schematisation.pick_cycles_point_number_as_df(ext, l_sig)
    cycles = schematisation.calc_cycles_parameters_by_numbers(ext, l_sig, cyc_num, traces, dt_max)
    cycles2 = schematisation.calc_cycles_parameters_by_numbers_2(ext, l_sig, cyc_num, [], dt_max)
    if code == 'MR':
        p1 = 'Mean'
        p2 = 'Range'
    else:
        p1 = 'Max'
        p2 = 'Min'
    tbls = schematisation.correlation_table_with_traces_2(cycles, p1, p2, traces,
                                                          mmin_set1=class_min2,
                                                          mmax_set1=class_max2, mmin_set2=class_min1,
                                                          mmax_set2=class_max1, count=m)
    data = [tbls[0].to_json(date_format='iso', orient='split')]
    for j in range(len(traces)):
        data.append(tbls[1][j].to_json(date_format='iso', orient='split'))
    dff = pd.DataFrame(data, columns=[code], index=['cycles'] + traces)
    return [dff, cycles2, ext]


def processing_parameters_set(flight, df, load_signals, k, hann, eps, traces, dt_max, class_min1, class_max1,
                              class_min2, class_max2, m, code='MR'):
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
    :param code: table code
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
    if True:
        # print(lengths[0])
        for i in range(lengths[0]):
            print(f"Processing signal: {load_signals[i]}")
            table, cycles, sig = processing_parameter(df, load_signals[i], k[i], hann[i], eps[i], traces[i], dt_max[i],
                                         class_min1[i], class_max1[i], class_min2[i], class_max2[i], m[i], code=code)
            # export table to folder load_signals[i] (parameter name)
            fname = load_signals[i] + '/' + flight + '.txt'
            table.to_csv(fname, index=True, encoding='utf-8')
            fname = load_signals[i] + '/' + flight + '-merged.txt'
            sig.to_csv(fname, index=False, encoding='utf-8')
            fname = load_signals[i] + '/' + flight + '-cycles.txt'
            cycles = get_full_cycles(cycles)
            cycles.to_csv(fname, index=False, encoding='utf-8')
        return "Processing Complete"
    else:
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


def get_signal_back(s):
    return s.replace(u"\u005c", '/')


def check_folders_tree(mode, sigs):
    s = ''
    folders = []
    isdir = os.path.exists(os.getcwd() + '/' + mode)
    if not isdir:
        s += f"Create folder {os.getcwd() + '/' + mode}\n"
    for sig in sigs:
        isdir = os.path.exists(os.getcwd() + '/' + mode + '/' + sig)
        folders.append(os.getcwd() + '/' + mode + '/' + sig + '/\n')
        if not isdir:
            s += f"Create folder {os.getcwd() + '/' + mode + '/' + sig}\n"

    Path(os.getcwd() + '/' + mode).mkdir(parents=True, exist_ok=True)
    [Path(os.getcwd() + '/' + mode + '/' + sig).mkdir(parents=True, exist_ok=True) for sig in sigs]
    return [s + "Folder tree is OK\n", folders]


def load_ave_files(filenames):
    # check if all codes are the same
    codes = set([read_data(fn, ['code'], ind=0)[1] for fn in filenames])
    if len(codes) != 1:
        return [{}, ["Error! Different codes in correlation tables."]]

    traces = set([len(read_data(fn, ['code'], ind=0)[0].index) for fn in filenames])
    if len(traces) != 1:
        return [{}, ["Error! Different traces in correlation tables."]]

    df, code = read_data(filenames[0], ['code'], ind=0)
    options = df.index.tolist()
    print('options={}'.format(options))
    data = {}
    counts_traces = {}
    classes = 1.0
    for opt in options:
        dfi = pd.read_json(df.loc[opt].values[0], orient='split')
        for col in dfi.columns:
            dfi[col] = pd.to_numeric(dfi[col], downcast='float')
        # print(dfi)
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

    # print(f"ct={counts_traces}")

    for i in range(1, len(filenames)):
        df, _ = read_data(filenames[i], ['code'], ind=0)
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

    # print(f"ct={counts_traces}")
    for opt in options:
        for i in range(classes):
            for j in range(classes):
                if counts_traces[opt][i][j] > 0:
                    # print(f"data={data[opt].values[i][j]}, c={counts_traces[opt][i][j]}")
                    val = float(data[opt].values[i][j]) / counts_traces[opt][i][j]
                    # print(f"val={val}")
                    data[opt]._set_value(data[opt].index[i], data[opt].columns[j], val)
                    # print(f"data={data[opt].values[i][j]}")

    data_str = {}
    for opt in options:
        data_str[opt] = data[opt].to_json(date_format='iso', orient='split')

    return [data_str, f"Load and average parameters: {options}"]


def convert_files(filenames):
    for filename in filenames:
        df, code = read_data(filename, ['code'], ind=0)
        st = loaddata.convert_corr_table_to_excel(df)
        f = open(filename.split('.')[0]+".dat", "w")
        f.write(st)
        f.close()
    return [f"Converting files: {filenames}. OK.", code]


def calc_kip(data, cut1):
    h = []
    h_r = []
    try:
        for key in data.keys():
            df = pd.read_json(data[key], orient='split')
            h_data = df.index[cut1 - 1]
            h.append(df.loc[h_data].to_numpy())
            h_r.append(df.columns.to_numpy())
        dff = schematisation.cumulative_frequency(h_r[0], h, list(data.keys()), False)
        return [dff, "Calculation complete. Create file kip.dat in current folder."]
    except:
        return [None, "Calculation failed."]


def export_kip(dir):
    print(dir)
    curr_dir = os.path.dirname(dir)
    os.chdir(curr_dir)
    _, _, filenames = next(walk(curr_dir))
    filenames = [file for file in filenames if '.txt' in file]
    print(f"Average files: {filenames}")
    df, status = load_ave_files(filenames)
    loading_data = json.dumps(df)
    print(status)
    status, code = convert_files(filenames)
    print(status)
    ave = pd.DataFrame(df.values(), columns=['code'], index=df.keys())
    # print(df.values())
    st = loaddata.convert_corr_table_to_excel(ave)
    f = open("average.dat", "w")
    f.write(st)
    f.close()
    print("Export average correlation table. OK.")

    try:
        csv_string = ''
        if 'cycles' in list(df.keys()):
            dff = pd.read_json(df['cycles'], orient='split')
            rows, cols = dff.shape
            # print(f"rows={rows}, cols={cols}")

            if rows * cols > 0:
                for i in range(1, rows - 1):
                    csv_string += loaddata.get_kpi(loading_data, i, code)

        f = open("kip.dat", "w")
        f.write(csv_string)
        return "Calculation complete. Convert files. Create files kip.dat and average.dat in current folder."
    except:
        return "Calculation failed."


def get_full_cycles(df):
    """
    Extract cycles with count=1.0 from cycles database and pick max-range cycle with count 0.5
    :param df: dataFrame of picking cycles
    :return: data frame with full cycles
    """
    dff = df.loc[df['Count'] == 1.0]
    half = df.loc[df['Count'] == 0.5]
    max_half = half.loc[half['Range'] == half['Range'].max()]
    dff = dff.append(max_half)
    dff.reset_index(drop=True, inplace=True)
    dff.drop(['Range', 'Mean'], axis=1, inplace=True)
    cols = ['Min', 'Max', 'Count', 't_min', 't_max']
    dff = dff[cols]
    dff['t_begin'] = dff.apply(lambda row: min(row.t_min, row.t_max), axis=1)
    dff = dff.sort_values(by=['t_begin'], ascending=True)
    dff.drop('t_begin', axis=1, inplace=True)
    print(dff)
    return dff

def convert_corr_table(data):
    pass