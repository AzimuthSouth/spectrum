from collections import defaultdict
import math
import numpy
import pandas
import matplotlib.pyplot as plt
import plotly.express as px


def merge(x, ind, d):
    # x - input, d - class width
    res = [x[0]]
    buf = []
    for i in range(1, len(x)):
        if len(buf) == 0:
            buf.append(res[- 1])

        if abs(buf[-1][ind] - x[i][ind]) <= d:
            buf.append(x[i])

        else:
            # move last res point to to buf
            buf = [res[-1]] + buf
            res = res[:-1]
            # average buf points
            ave = average_array(buf)
            res.append(ave)
            res.append(x[i])
            buf = []

        if i == (len(x) - 1):
            if abs(res[-1][ind] - x[i][ind]) <= d:
                res = res[:-1]
            res.append(x[i])

    return numpy.array(res)


def average_array(buf):
    res = numpy.zeros(len(buf[0]))
    for i in range(len(buf)):
        for j in range(len(buf[0])):
            res[j] += buf[i][j]
    res = numpy.array(res) / len(buf)
    return res


# detect if a point is an extreme of the series
def is_extreme(x):
    # print(is_max(x) or is_min(x))
    return is_max(x) or is_min(x)


def is_max(x):
    if (x[0] < x[1]) and (x[1] > x[2]):
        return True
    return False


def is_min(x):
    if (x[0] > x[1]) and (x[1] < x[2]):
        return True
    return False


###


def extreme_count(x):
    # first and last points are always extremes
    res = 2
    for i in range(1, len(x) - 1):
        if is_extreme(x[i - 1:i + 2]):
            res += 1
    return res


# class width, if data series is divided by m classes
def class_width(x, m):
    delta = (numpy.max(x) - numpy.min(x)) / m
    return delta


def set_classes(x, m):
    res = []
    for i in range(m):
        res.append(numpy.min(x[:, 1]) + i * class_width(x[:, 1], m))
    res.append(numpy.max(x[:, 1]))
    return numpy.array(res)


def set_n_segments(x, dt):
    # choose segments lasting dt, add previous and next time points for extreme detection
    xl = x.tolist()
    seg = []
    ind = 1
    buf = [xl[ind - 1]]
    for i in range(1, len(x[:, 0] - 1)):
        if abs(x[i, 0] - x[ind, 0]) < dt:
            buf.append(xl[i])
        else:
            buf.append(xl[i])
            seg.append(numpy.array(buf))
            ind = i
            buf = [xl[ind - 2], xl[ind - 1], xl[i]]
    seg.append(numpy.array(buf))
    return numpy.array(seg, dtype=numpy.ndarray)


def max_frequency(x, n):
    dt = (numpy.max(x[:, 0]) - numpy.min(x[:, 0])) / n
    seg = set_n_segments(x, dt)
    f = []
    for i in range(len(seg)):
        dti = seg[i][-2, 0] - seg[i][1, 0]
        f.append(extreme_count(seg[i][:, 1]) / (2 * dti))
    return numpy.max(f)


# number of mid-level crossings
def mean_count(x):
    res = 0
    mn = numpy.mean(x)
    print(mn)
    for i in range(1, len(x)):
        if (x[i - 1] < mn) and (x[i] > mn):
            res += 1
        if (x[i - 1] > mn) and (x[i] < mn):
            res += 1
    return res


def input_stats(data):
    mn = numpy.mean(data[:, 1])
    s2 = numpy.var(data[:, 1])
    st = numpy.std(data[:, 1])
    kp = mean_count(data[:, 1]) / extreme_count(data[:, 1])
    return [mn, s2, st, kp]


def pick_extremes(data, ind):
    """
    pick extremes from data series,
    1st ans last points are extremes
    :param data: data array
    :param ind: column for pick extremes
    :return: array of selected rows
    """
    x = data[:, ind]
    res = [data[0]]
    for i in range(1, len(data) - 1):
        if is_extreme(x[i - 1:i + 2]):
            res.append(data[i])
    res.append(data[-1])
    return numpy.array(res)


# schematization of data series by extremes method
def extremes_method(data, ind, m):
    ext = pick_max_above0_min_below0(data, ind)
    return repetition_rate(ext[:, ind], count=m)


def maximum_method(data, ind, m):
    ext = pick_max_above0(data, ind)
    return repetition_rate(ext[:, ind], count=m)


def minimum_method(data, ind, m):
    ext = pick_min_below0(data, ind)
    return repetition_rate(ext[:, ind], count=m)


def range_method(data, ind, m):
    return repetition_rate(pick_ranges(data, ind), m)


# helpers for schematisation methods
def pick_max_above0(data, ind):
    x = data(data[:, ind])
    res = []
    for i in range(1, len(x) - 1):
        if is_max(x[i - 1: i + 2]) and (x[i] > 0):
            res.append(data[i])
    return res


def pick_min_below0(data, ind):
    x = data(data[:, ind])
    res = []
    for i in range(1, len(x) - 1):
        if is_min(x[i - 1: i + 2]) and (x[i] < 0):
            res.append(data[i])
    return res


def pick_max_above0_min_below0(data, ind):
    return pick_max_above0(data, ind) + pick_min_below0(data, ind)


def pick_ranges(data, ind):
    n2 = numpy.ceil(len(data) / 2)
    res = []
    for i in range(n2):
        p1 = data[2 * i]
        p2 = data[2 * i + 1]
        res.append(abs(p1[ind] - p2[ind]))
    return res


# calc 1D repetition rate
def repetition_rate(data, width=None, count=None):
    counts = defaultdict()
    if count is not None:
        width = (max(data) - min(data)) / count

    if width is not None:
        nmax = 0
        for x in data:
            n = int(math.ceil(x / width))
            counts[n * width] += 1.0
            nmax = max(n, nmax)

        for i in range(1, nmax):
            counts.setdefault(i * width, 0.0)

    return sorted(counts.items())


# calc correlation table
def correlation_table(cycles, name1, name2, count=10):
    """
    Calc correlation table min-max or mean-range for loading input
    :param cycles: dataFrame with extracted half-cycles
    :param count:  class numbers
    :param name1: variable 1 (min or mean)
    :param name2: variable 2 (max or range)
    :return: dataFrame with correlation table
    """
    rows, _ = cycles.shape
    # set classes width
    print("Min: from={}, to={}".format(cycles[name1].max(), cycles[name1].min()))
    print("Max: from={}, to={}".format(cycles[name2].max(), cycles[name2].min()))
    mmin = min(cycles[name1].min(), cycles[name2].min())
    mmax = max(cycles[name1].max(), cycles[name2].max())
    print("min={}, max={}".format(mmin, mmax))
    w = (mmax - mmin) / (count - 1)

    print("w={}".format(w))
    # set classes names
    name_rows = []
    name_cols = []
    for i in range(count):
        r1 = mmin + w * i - w / 2
        r2 = mmin + w * i + w / 2
        r = mmin + w * i
        # name_rows.append("{:.3f}-{:.3f}".format(r1, r2))
        # name_cols.append("{:.3f}-{:.3f}".format(r1, r2))
        name_rows.append("{:.3f}".format(r))
        name_cols.append("{:.3f}".format(r))
    print(name_rows)
    print(name_cols)
    res = numpy.zeros((count, count))
    for i in range(rows):
        ind1 = int(math.trunc(cycles.iloc[i][name1] / w)) - 1
        ind2 = int(math.trunc(cycles.iloc[i][name2] / w)) - 1
        # print("i={}, j={}, ind1={}, ind2={}".format(cycles.iloc[i][name1], cycles.iloc[i][name2], ind1, ind2))
        res[ind1][ind2] += cycles.iloc[i]['Count']

    df = pandas.DataFrame(res, columns=name_cols, index=name_rows)
    df.style.background_gradient(cmap='Blues', axis=None)
    plt.pcolor(df, cmap='GnBu')
    plt.yticks(numpy.arange(0.5, len(df.index), 1), df.index, fontsize=10)
    plt.xticks(numpy.arange(0.5, len(df.columns), 1), df.columns, fontsize=10, rotation=90)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=10)
    plt.show()
    return df


# calc range and mean values for cycle
def cycle_parameters(x1, x2, c):
    # return range, count, mean, min, max
    return [abs(x1 - x2), c, (x1 + x2) / 2, min(x1, x2), max(x1, x2)]


def _get_round_function(ndigits=None):
    if ndigits is None:
        def func(x):
            return x
    else:
        def func(x):
            return round(x, ndigits)
    return func


# extract cycles from extreme array
def pick_cycles(data):
    stack = []
    res = []
    x = data[:, 1]

    for i in range(len(data)):
        stack.append(x[i])

        while len(stack) >= 3:
            [a, b, c] = stack[-3:]
            ab = abs(a - b)
            bc = abs(b - c)

            if bc < ab:
                # read next point
                break
            elif len(stack) == 3:
                # left half-cycle
                res.append(cycle_parameters(a, b, 0.5))
                stack = stack[1:]
            else:
                # left cycle
                res.append(cycle_parameters(a, b, 1))
                last = stack[-1]
                stack = stack[:-3]
                stack.append(last)
    else:
        while len(stack) > 1:
            res.append(cycle_parameters(stack[0], stack[1], 0.5))
            stack = stack[1:]

    return res


def pick_cycles_as_df(data):
    res = pick_cycles(data)
    df = pandas.DataFrame(res, columns=['Range', 'Count', 'Mean', 'Min', 'Max'])
    return df


def count_cycles(data, ndigits=None, n_seg=None, seg_size=None):
    counts = defaultdict(float)

    print(numpy.array(pick_cycles(data)))

    cycles = [i[:2] for i in pick_cycles(data)]

    if n_seg is not None:
        seg_size = (max(data[:, 1]) - min(data[:, 1])) / n_seg

    if seg_size is not None:
        nmax = 0
        for rng, count in cycles:
            n = int(math.ceil(rng / seg_size))  # using int for Python 2 compatibility
            counts[n * seg_size] += count
            nmax = max(n, nmax)

        for i in range(1, nmax):
            counts.setdefault(i * seg_size, 0.0)

    elif ndigits is not None:
        round_ = _get_round_function(ndigits)
        for rng, count in cycles:
            counts[round_(rng)] += count

    else:
        for rng, count in cycles:
            counts[rng] += count

    return sorted(counts.items())


def rain_flow2(data):
    """
    3 point rainflow algorithm to extract means and ranges
    :param data : [[t, input(t)],..]:
    :return: [cycle_range, point1, point2]
    """
    x = data[:, 1]
    stack = []
    res = []
    [a, b, c] = x[:3]
    ind = 3
    mem = 0
    b_flag = False
    c_flag = False
    end_flag = False

    while ind <= len(data):

        if (ind == len(x)) and (mem == 0):
            end_flag = True

        if end_flag:
            break

        ab = abs(b - a)
        bc = abs(c - b)

        if ab == bc:
            res.append([ab, b, a])
            if mem == 0:
                a = c
                b_flag = True
                c_flag = True
            elif mem == 1:
                a = stack[-1]
                b = c
                mem -= 1
                b_flag = False
                c_flag = True
            elif mem >= 2:
                a = stack[-2]
                b = stack[-1]
                mem -= 2
                b_flag = False
                c_flag = False
        elif bc > ab:
            res.append([ab, b, a])
            if mem == 0:
                a = b
                b = c
                b_flag = False
                c_flag = True
            elif mem == 1:
                a = stack[-1]
                b = c
                mem -= 1
                b_flag = False
                c_flag = True
            elif mem >= 2:
                a = stack[-2]
                b = stack[-1]
                mem -= 2
                b_flag = False
                c_flag = False
        elif bc < ab:
            mem += 1
            stack.append(a)
            a = b
            b = c
            b_flag = False
            c_flag = True

        if b_flag:
            b = x[ind]
            ind += 1
            b_flag = False

        if c_flag:
            c = x[ind]
            ind += 1
            c_flag = False

    res.append([abs(b - a), b, a])
    print(res)
