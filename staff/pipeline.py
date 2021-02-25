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
