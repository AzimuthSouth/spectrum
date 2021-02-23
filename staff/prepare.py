import numpy


def smoothing_symm(data, name, k, step):
    """
    Function for smoothing and thinning input signal
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
    Function for smoothing and thinning input signal
    :param data: dataframe
    :param names: array of col_names for smoothing
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


def correction_hann(data, name):
    """

    :param data: dataFrame
    :param name: column name for correction
    :return: dataFrame
    """
    rows, _ = data.shape
    dff = data.copy()
    xi = [hann_coefficient(i, rows) for i in range(rows)]
    dff[name] *= xi
    return dff


def set_correction_hann(data, names):
    """

    :param data: dataFrame
    :param names: array of column name for correction
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
        dff[name + "_hann"] = dff[name] * xi
    return dff


def calc_time_range(time):
    delta = abs(time[:-1] - time[1:])
    return [min(time), max(time), numpy.mean(delta), numpy.std(delta)]
