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
        if smoothing > 0:
            sig = prepare.smoothing_symm(df, column, smoothing, 1)
        _, g_xy = signal.csd(sig[column], sig[column], (1.0 / dt), window=win, nperseg=rows)
        name = column + "_" + column
        dff[name] = g_xy
    for i in range(1, len(col_names)):
        for j in range(i + 1, len(col_names)):
            if smoothing > 0:
                sig = prepare.smoothing_symm(df, col_names[i], smoothing, 1)
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
            sig1 = df[col_names[i]]
            sig2 = df[col_names[j]]
            if smoothing > 0:
                sig = prepare.smoothing_symm(df, col_names[i], smoothing, 1)
                sig = prepare.smoothing_symm(sig, col_names[j], smoothing, 1)
            _, c_xy = signal.coherence(sig[col_names[i]], sig[col_names[j]],
                                       (1.0 / dt), window=win, nperseg=npseg)
            name = col_names[i] + "_" + col_names[j]
            dff[name] = c_xy
    return dff


def signal_processing(df, name, eps):
    """

    :param df: input signals dataFrame
    :param name: expected signal column name
    :param eps: class width
    :return: repetition rate
    """
    # pick signal
    time = df[df.columns[0]]
    sig = df[name]

    # merge signal and pick extremes
    sig_merge = schematisation.merge(zip(time, sig), 1, eps)
    sig_ext = schematisation.pick_extremes(sig_merge, 1)

    # there should be detrend!!!

    # initial statistics
    stats = schematisation.input_stats(sig_ext)


