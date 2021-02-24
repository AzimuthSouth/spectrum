import pandas as pd
from staff import prepare
from staff import analyse
from staff import schematisation
from scipy import signal
import scipy


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
        name = column + "_" + column
        dff[name] = g_xy
    for i in range(1, len(col_names)):
        for j in range(i + 1, len(col_names)):
            sig = df.copy()
            if smoothing > 0:
                sig = prepare.smoothing_symm(sig, col_names[i], smoothing, 1)
                sig = prepare.smoothing_symm(sig, col_names[j], smoothing, 1)
            _, g_xy = signal.csd(sig[col_names[i]], sig[col_names[j]], (1.0 / dt), window=win, nperseg=rows)
            name = col_names[i] + "_" + col_names[j]
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
    if npseg > rows:
        npseg = rows
    if smoothing > 0:
        rows -= smoothing
    f = scipy.fft.rfftfreq(npseg, dt)
    dff["Frequencies"] = f
    for i in range(1, len(col_names)):
        for j in range(i + 1, len(col_names)):
            sig = df.copy()
            if smoothing > 0:
                sig = prepare.smoothing_symm(df, col_names[i], smoothing, 1)
                sig = prepare.smoothing_symm(sig, col_names[j], smoothing, 1)
            _, c_xy = signal.coherence(sig[col_names[i]], sig[col_names[j]],
                                       (1.0 / dt), window=win, nperseg=npseg)
            name = col_names[i] + "_" + col_names[j]
            dff[name] = c_xy
    return dff


def signal_processing(df, name, eps, code):
    """

    :param df: input signals dataFrame
    :param name: expected signal column name
    :param eps: class width
    :param code: 1 - for min/max correlation table, 2 - for ave/range correlation table
    :return: repetition rate
    """
    # pick signal
    time = df[df.columns[0]].to_numpy()
    sig = df[name].to_numpy()
    # merge signal and pick extremes
    sig_merge = schematisation.merge(list(zip(time, sig)), 1, eps)
    sig_ext = schematisation.pick_extremes(sig_merge, 1)
    print("extremes = {}".format(sig_ext))
    # there should be detrend??

    # initial statistics
    stats = schematisation.input_stats(sig_ext)
    print('stats={}'.format(stats))

    # pick cycles
    cycles = schematisation.pick_cycles_as_df(sig_ext)
    print("cycles = {}".format(cycles))
    tbl = pd.DataFrame()

    if code == 1:
        tbl = schematisation.correlation_table(cycles, 'Max', 'Min', 10)
    if code == 2:
        tbl = schematisation.correlation_table(cycles, 'Range', 'Mean', 10)

    print(tbl)

    return tbl

