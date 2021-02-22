import numpy


def smoothing_symm(data, name, k, step):
    """
    Function for smoothing and offsetting input signal
    :param data: dataframe
    :param name: col_name for smoothing
    :param k: window width
    :param step: offset
    :return: dataFrame
    """
    x = data[name].to_numpy()
    x_smooth = []
    # 1st corrected point
    ind1 = int(numpy.trunc(k / 2))
    # select rows from df
    index = range(ind1, len(x) - ind1, step)
    dff = data.iloc[index]
    dff.reset_index(drop=True, inplace=True)
    # smooth column
    for i in range(ind1, len(x) - ind1, step):
        xi = numpy.mean(x[i: k + i])
        x_smooth.append(xi)
    df1 = dff.copy()
    df1[name] = x_smooth
    return df1

def set_smoothing_symm(data, names, k, step):
    """
    Function for smoothing and offsetting input signal
    :param data: dataframe
    :param name: array of col_names for smoothing
    :param k: window width
    :param step: offset
    :return: dataFrame
    """
    rows, _ = data.shape
    # 1st corrected point
    ind1 = int(numpy.trunc(k / 2))
    # select rows from df
    index = range(ind1, rows - ind1, step)
    dff = data.iloc[index]
    dff.reset_index(drop=True, inplace=True)
    df1 = dff.copy()

    for name in names:
        x = data[name].to_numpy()
        x_smooth = []
        # smooth column
        for i in range(ind1, len(x) - ind1, step):
            xi = numpy.mean(x[i: k + i])
            x_smooth.append(xi)
        df1[name + '_smooth'] = x_smooth
    return df1


def smoothing2(data, k, step):
    res = []
    # mn = numpy.mean(data)
    for i in range(len(data) - k + 1):
        xi = data[i:k + i]
        res.append(numpy.mean(xi))
    return res


def hann_coefficient(r, n):
    return (1 - numpy.cos(2 * r * numpy.pi / n)) / 2


def triangle_coefficient(r, n):
    return 1 - 2 / n * abs(r - n / 2)


def hemming_coefficient(r, n):
    return 0.54 + 0.46 * numpy.cos(2 * numpy.pi * ((2 * r - n) / 2 / n))


def natoll_coefficient(r, n, a1, a2, a3):
    ti = 2 * numpy.pi * ((2 * r - n) / 2 / n)
    return a1 * numpy.cos(ti) + a2 * numpy.cos(2 * ti) + a3 * numpy.cos(3 * ti)


def correction_hann(data, name, par=None):
    """

    :param data: dataFrame
    :param name: column name for correction
    :param par:
    :return: dataFrame
    """
    rows, _ = data.shape
    dff = data.copy()
    xi = [hann_coefficient(i, rows) for i in range(rows)]
    dff[name] *= xi
    return dff


def set_correction_hann(data, names, par=None):
    """

    :param data: dataFrame
    :param names: array of column name for correction
    :param par:
    :return: dataFrame
    """
    col_names = data.columns
    # check if smoothing signals exists
    smooth = False
    for col in col_names:
        if "_smooth" in col:
            smooth = True
            break
    rows, _ = data.shape
    dff = data.copy()

    xi = [hann_coefficient(i, rows) for i in range(rows)]
    for name in names:
        if smooth:
            name += "_smooth"
        dff[name + "hann"] = dff[name] * xi
    return dff


def correction_triangle(data, par=None):
    res = []
    for i in range(len(data)):
        res.append(data[i] * triangle_coefficient(i, len(data) - 1))
    return res


def correction_hemming(data, par=None):
    res = []
    for i in range(len(data)):
        res.append(data[i] * hemming_coefficient(i, len(data) - 1))
    return res


def correction_natoll(data, par):
    res = []
    for i in range(len(data)):
        res.append(data[i] * natoll_coefficient(i, len(data) - 1, par[0], par[1], par[2]))
    return res


def correction(data, code, par=None):
    if code == 1:
        return correction_triangle(data, par)
    if code == 2:
        return correction_hann(data, par)
    if code == 3:
        return correction_hemming(data, par)
    if code == 4:
        return correction_natoll(data, par)


def calc_time_range(time):
    delta = abs(time[:-1] - time[1:])
    return [min(time), max(time), numpy.mean(delta), numpy.std(delta)]
