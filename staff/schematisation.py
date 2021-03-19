from collections import defaultdict
import math
import numpy
import pandas


def merge(df, name, d):
    """

    :param df: dataFrame
    :param name: merge signal
    :param d: merge width
    :return: dataFrame
    """
    # x - input, d - class width
    col_names = df.columns
    # x = list(zip(df[col_names[0]].to_numpy(), df[name].to_numpy()))
    x = df.to_numpy()
    y = df[name].to_numpy()
    res = [x[0]]
    res_y = [y[0]]
    buf = []
    buf_y = []
    ind = 1
    for i in range(1, len(x)):
        if len(buf) == 0:
            buf.append(res[- 1])
            buf_y.append(res_y[-1])
        if abs(buf_y[-1] - y[i]) <= d:
            buf.append(x[i])
            buf_y.append(y[i])

        else:
            res = res[:-1]
            res_y = res_y[:-1]
            # average buf points
            ave = average_array(buf)
            ave_y = numpy.mean(buf_y)
            res.append(ave)
            res.append(x[i])
            buf = []
            res_y.append(ave_y)
            res_y.append(y[i])
            buf_y = []

        if i == (len(x) - 1):
            if abs(res_y[-1] - y[i]) <= d:
                res = res[:-1]
                res_y = res_y[:-1]
            res.append(x[i])
            res_y.append(y[i])
    dff = pandas.DataFrame(res, columns=col_names)
    return dff


def merge_extremes(df, name, d):
    """

    :param df: dataFrame
    :param name: merge signal
    :param d: merge width
    :return: dataFrame
    """
    col_names = df.columns
    x = df.to_numpy()
    y = df[name].to_numpy()
    # choose 1st point
    ind = 1
    i = 0
    while (abs(y[i] - y[i + 1]) < d) and (abs(y[i] - y[0]) < d):
        i += 1
    if i == 0:
        res = [x[0], x[1]]
        res_y = [y[0], y[1]]
        i += 1
    else:
        if abs(y[0] - y[i]) >= d:
            res = [x[0], x[i]]
            res_y = [y[0], y[i]]
        else:
            res = [x[i]]
            res_y = [y[i]]
    curr_is_min = is_min(y[i - 1: i + 2])
    # print("input res={}".format(res))
    for j in range(i + 1, len(x)):
        change_brunch = True
        last = res_y[-1]
        if curr_is_min and (y[j] < last):
            res = res[:-1]
            res_y = res_y[:-1]
            res.append(x[j])
            res_y.append(y[j])
            change_brunch = False
        if (not curr_is_min) and (y[j] > last):
            res = res[:-1]
            res_y = res_y[:-1]
            res.append(x[j])
            res_y.append(y[j])
            change_brunch = False
        # print("change={}, d={}, y={}, res={}".format(change_brunch, abs(res[-1][ind] - x[j][ind]), x[j][ind], res))
        if change_brunch:
            if abs(res_y[-1] - y[j]) >= d:
                res.append(x[j])
                res_y.append(y[j])
                curr_is_min = not curr_is_min

    dff = pandas.DataFrame(res, columns=col_names)
    return dff


def get_merged_extremes(df, name, d):
    remove_steps = merge(df, name, 0)
    extremes = pick_extremes(remove_steps, name)
    if d is None:
        pass
    else:
        extremes = merge_extremes(all_extremes, name, d)
    return extremes


def get_extremes(df, name):
    remove_steps = merge(df, name, 0)
    all_extremes = pick_extremes(remove_steps, name)
    return all_extremes


def average_array(buf):
    res = numpy.zeros(len(buf[0]))
    for i in range(len(buf)):
        for j in range(len(buf[0])):
            res[j] += buf[i][j]
    res = numpy.array(res) / len(buf)
    return res


# detect if a point is an extreme of the series
def is_extreme(x):
    return is_max(x) or is_min(x)


def is_max(x):
    if (x[0] < x[1]) and (x[1] > x[2]):
        return True
    return False


def is_min(x):
    if (x[0] > x[1]) and (x[1] < x[2]):
        return True
    return False


def extreme_count(x):
    # first and last points are always extremes
    res = 2
    for i in range(1, len(x) - 1):
        if is_extreme(x[i - 1:i + 2]):
            res += 1
    return res


'''
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
'''


def f_estimate(count, dt):
    return count / 2 / dt


def max_frequency(df, name, n, d=None):
    """
    Max frequency estimation
    :param df: dataFrame input signal
    :param name: column name
    :param n: segments for extremes counting
    :param d: amplitude filter width
    :return: float, frequency estimation
    """
    if n is None:
        pass
    else:
        # get extremes
        if d is None:
            extremes = get_extremes(df, name)
        else:
            extremes = get_merged_extremes(df, name, d)

        rows, _ = extremes.shape
        col_names = extremes.columns
        f_estimation = []
        time = extremes[col_names[0]].to_numpy()[1: -1]
        dt = (max(time) - min(time)) / n
        # print('extr={}, dt={}'.format(extremes, dt))
        ind = 1
        t_start = extremes[col_names[0]].iloc[1]
        for i in range(n):
            count = 1
            ti_start = extremes[col_names[0]].iloc[ind]
            ti_curr = extremes[col_names[0]].iloc[ind]
            while (ti_curr <= t_start + dt * (i + 1)) and (ind < rows):
                ind += 1
                count += 1
                ti_curr = extremes[col_names[0]].iloc[ind]
            ti_curr = extremes[col_names[0]].iloc[ind - 1]
            # print('ti_curr={}, ti_start={}, count={}'.format(ti_curr, ti_start, count))
            if ti_curr > ti_start:
                f_estimation.append(f_estimate(count - 2, ti_curr - ti_start))
        if f_estimation:
            return max(f_estimation)
        return "Too few extremes for frequency estimation"


# number of mid-level crossings
def mean_count(x):
    res = 0
    mn = numpy.mean(x)
    if x[0] == mn:
        res += 1
    if x[-1] == mn:
        res += 1
    for i in range(1, len(x)):
        if (x[i - 1] < mn) and (x[i] > mn):
            res += 1
        if (x[i - 1] > mn) and (x[i] < mn):
            res += 1
    return res


def input_stats(df, name):
    """
    input statistic for signal
    :param df: dataFrame
    :param name: column name
    :return: mean, variance, standard deviation, reg.coefficient
    """
    data = df[name].to_numpy()
    mn = numpy.mean(data)
    s2 = numpy.var(data)
    st = numpy.std(data)
    kp = mean_count(data) / extreme_count(data)
    return [mn, s2, st, kp]


def pick_extremes(df, name):
    """
    pick extremes from data series,
    1st ans last points are extremes
    :param df: dataFrame
    :param name: column for pick extremes
    :return: dataFrame with df[name - extremes, other columns from same rows]
    """
    col_names = df.columns
    x = df[name].to_numpy()
    data = df.to_numpy()
    res = [data[0]]
    for i in range(1, len(data) - 1):
        if is_extreme(x[i - 1:i + 2]):
            res.append(data[i])
    res.append(data[-1])
    dff = pandas.DataFrame(res, columns=col_names)
    return dff


'''
# schematisation of data series by extremes method
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

'''


# calc 1D repetition rate
def repetition_rate(data, width=None, count=None):
    """

    :param data: 1D array of data to calc repetition rate of classes
    :param width: class width
    :param count: classes count
    :return: dictionary of classes and rates
    """
    counts = defaultdict()

    if width is not None:
        count = int(math.ceil((max(data) - min(data)) / width))
    else:
        width = (max(data) - min(data)) / count

    for i in range(count):
        counts.setdefault((i + 1) * width, 0.0)
    if width is not None:
        for x in data:
            n = int(math.trunc(x / width))
            if n == count:
                n -= 1
            # print('x={}, n={}'.format(x, n))
            counts[(n + 1) * width] += 1.0

    return sorted(counts.items())


# calc correlation table
def correlation_table(cycles, name1, name2, mmin_set=None, mmax_set=None, count=10):
    """
    Calc correlation table min-max or mean-range for loading input
    :param cycles: dataFrame with extracted half-cycles
    :param count:  class numbers
    :param name1: variable 1 (min or mean)
    :param name2: variable 2 (max or range)
    :param mmin_set: classes minimum
    :param mmax_set: classes maximum
    :return: dataFrame with correlation table
    """
    rows, _ = cycles.shape
    # set classes width
    mmin = min(cycles[name1].min(), cycles[name2].min()) if mmin_set is None else mmin_set
    mmax = max(cycles[name1].max(), cycles[name2].max()) if mmax_set is None else mmax_set
    w = (mmax - mmin) / count

    # set classes names
    name_rows = []
    name_cols = []
    for i in range(count):
        r1 = mmin + w * i
        r2 = mmin + w * (i + 1)
        r = mmin + w * i
        name_rows.append("{:.2f}-{:.2f}:_{}".format(r1, r2, i + 1))
        name_cols.append("{}".format(i + 1))
        # name_rows.append("{:.2f}".format(r2))
        # name_cols.append("{:.2f}".format(r2))
    res = numpy.zeros((count, count))
    if w == 0:
        res += cycles.iloc[0]['Count']
    for i in range(rows):
        ind1 = 0
        ind2 = 0
        if w > 0:
            ind1 = int(math.trunc((cycles.iloc[i][name1] - mmin) / w))
            ind2 = int(math.trunc((cycles.iloc[i][name2] - mmin) / w))
            if ind1 == count:
                ind1 -= 1
            if ind2 == count:
                ind2 -= 1
            # print("x={}, y={}, ind1={}, ind2={}".format(cycles.iloc[i][name1], cycles.iloc[i][name2], ind1, ind2))
        if is_between(ind1, 0, count) and is_between(ind2, 0, count):
            res[ind1][ind2] += cycles.iloc[i]['Count']

    df = pandas.DataFrame(res, columns=name_cols, index=name_rows)
    return df


# calc correlation table
def correlation_table_with_traces(cycles, name1, name2, traces_names=[], mmin_set=None, mmax_set=None, count=10):
    """
    Calc correlation table min-max or mean-range for loading input
    :param cycles: dataFrame with extracted half-cycles
    :param count:  class numbers
    :param name1: variable 1 (min or mean)
    :param name2: variable 2 (max or range)
    :param traces_names: additional signals
    :param mmin_set: classes minimum
    :param mmax_set: classes maximum
    :return: dataFrame with correlation table
    """
    rows, _ = cycles.shape
    # set classes width
    mmin = min(cycles[name1].min(), cycles[name2].min()) if mmin_set is None else mmin_set
    mmax = max(cycles[name1].max(), cycles[name2].max()) if mmax_set is None else mmax_set
    w = (mmax - mmin) / count

    # set classes names
    name_rows = []
    name_cols = []
    for i in range(count):
        r1 = mmin + w * i
        r2 = mmin + w * (i + 1)
        r = mmin + w * i
        name_rows.append("{:.2f}-{:.2f}:     {}".format(r1, r2, i + 1))
        name_cols.append("{}".format(i + 1))
        # name_rows.append("{:.2f}".format(r2))
        # name_cols.append("{:.2f}".format(r2))
    res = numpy.zeros((count, count))
    traces = [numpy.empty((count, count), dtype=list) for i in range(len(traces_names))]
    traces_mean = [numpy.zeros((count, count)) for i in range(len(traces_names))]
    if w == 0:
        res += cycles.iloc[0]['Count']
    for i in range(rows):
        ind1 = 0
        ind2 = 0
        if w > 0:
            ind1 = int(math.trunc((cycles.iloc[i][name1] - mmin) / w))
            ind2 = int(math.trunc((cycles.iloc[i][name2] - mmin) / w))
            if ind1 == count:
                ind1 -= 1
            if ind2 == count:
                ind2 -= 1
            # print("x={}, y={}, ind1={}, ind2={}".format(cycles.iloc[i][name1], cycles.iloc[i][name2], ind1, ind2))
        if is_between(ind1, 0, count) and is_between(ind2, 0, count):
            res[ind1][ind2] += cycles.iloc[i]['Count']
            for j in range(len(traces_names)):
                if cycles[traces_names[j]][i] == numpy.inf:
                    pass
                else:
                    if traces[j][ind1][ind2] is None:
                        traces[j][ind1][ind2] = []
                    traces[j][ind1][ind2].append(cycles[traces_names[j]][i])
    for j in range(len(traces_names)):
        for i in range(count):
            for k in range(count):
                if traces[j][i][k] is None:
                    pass
                else:
                    traces_mean[j][i][k] = numpy.mean(traces[j][i][k])
    df_traces = []
    df = pandas.DataFrame(res, columns=name_cols, index=name_rows)
    for j in range(len(traces_names)):
        dff = pandas.DataFrame(traces_mean[j], columns=name_cols, index=name_rows)
        df_traces.append(dff)
    return [df, df_traces]


def correlation_table_with_traces_2(cycles, name1, name2, traces_names=[], mmin_set1=None, mmax_set1=None,
                                    mmin_set2=None, mmax_set2=None, count=10):
    """
    Calc correlation table min-max or mean-range for loading input
    :param cycles: dataFrame with extracted half-cycles
    :param count:  class numbers
    :param name1: variable 1 (min or mean)
    :param name2: variable 2 (max or range)
    :param traces_names: additional signals
    :param mmin_set1: classes minimum
    :param mmax_set1: classes maximum
    :param mmin_set2: classes minimum
    :param mmax_set2: classes maximum
    :return: dataFrame with correlation table
    """
    rows, _ = cycles.shape
    # set classes width
    mmin1 = cycles[name1].min() if mmin_set1 is None else mmin_set1
    mmax1 = cycles[name1].max() if mmax_set1 is None else mmax_set1
    mmin2 = cycles[name2].min() if mmin_set2 is None else mmin_set2
    mmax2 = cycles[name2].max() if mmax_set2 is None else mmax_set2

    w1 = (mmax1 - mmin1) / count
    w2 = (mmax2 - mmin2) / count

    # set classes names
    name_rows = []
    name_cols = []
    for i in range(count):
        r11 = mmin1 + w1 * i
        r21 = mmin1 + w1 * (i + 1)
        r12 = mmin2 + w2 * i
        r22 = mmin2 + w2 * (i + 1)

        name_rows.append("{:.2f}-{:.2f}".format(r11, r21))
        name_cols.append("{:.2f}-{:.2f}".format(r12, r22))
        # name_rows.append("{:.2f}".format(r2))
        # name_cols.append("{:.2f}".format(r2))
    res = numpy.zeros((count, count))
    traces = [numpy.empty((count, count), dtype=list) for i in range(len(traces_names))]
    traces_mean = [numpy.zeros((count, count)) for i in range(len(traces_names))]
    if w1 == 0 or w2 ==0:
        res += cycles.iloc[0]['Count']
    for i in range(rows):
        ind1 = 0
        ind2 = 0
        if w1 > 0 and w2 > 0:
            ind1 = int(math.trunc((cycles.iloc[i][name1] - mmin1) / w1))
            ind2 = int(math.trunc((cycles.iloc[i][name2] - mmin2) / w2))
            if ind1 == count:
                ind1 -= 1
            if ind2 == count:
                ind2 -= 1
            # print("x={}, y={}, ind1={}, ind2={}".format(cycles.iloc[i][name1], cycles.iloc[i][name2], ind1, ind2))
        if is_between(ind1, 0, count) and is_between(ind2, 0, count):
            res[ind1][ind2] += cycles.iloc[i]['Count']
            for j in range(len(traces_names)):
                if cycles[traces_names[j]][i] == numpy.inf:
                    pass
                else:
                    if traces[j][ind1][ind2] is None:
                        traces[j][ind1][ind2] = []
                    traces[j][ind1][ind2].append(cycles[traces_names[j]][i])
    for j in range(len(traces_names)):
        for i in range(count):
            for k in range(count):
                if traces[j][i][k] is None:
                    pass
                else:
                    traces_mean[j][i][k] = numpy.mean(traces[j][i][k])
    df_traces = []

    df = pandas.DataFrame(res, columns=name_cols, index=name_rows)
    for j in range(len(traces_names)):
        dff = pandas.DataFrame(traces_mean[j], columns=name_cols, index=name_rows)
        df_traces.append(dff)
    return [df, df_traces]



def is_between(x, mn, mx):
    return (x >= mn) and (x <= mx)


# calc range and mean values for cycle
def cycle_parameters(x1, x2, c):
    # return range, count, mean, min, max
    return [abs(x1 - x2) / 2, c, (x1 + x2) / 2, min(x1, x2), max(x1, x2)]


def cycle_point_numbers(x1, x2, c):
    # return number1, number2, count
    min_point = x1[0] if x1[1] < x2[1] else x2[0]
    max_point = x1[0] if x1[1] > x2[1] else x2[0]
    return [min_point, max_point, c]


def pick_cycles(df, name):
    """

    :param df: dataFrame
    :param name: column name
    :return:
    """
    stack = []
    res = []
    x = df[name].to_numpy()

    for i in range(len(x)):
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


def pick_cycles_point_numbers(df, name):
    """

    :param df: dataFrame
    :param name: column name
    :return:
    """
    stack = []
    res = []
    x = df[name].to_numpy()

    for i in range(len(x)):
        stack.append([i, x[i]])

        while len(stack) >= 3:
            [a, b, c] = stack[-3:]
            ab = abs(a[1] - b[1])
            bc = abs(b[1] - c[1])

            if bc < ab:
                # read next point
                break
            elif len(stack) == 3:
                # left half-cycle
                res.append(cycle_point_numbers(a, b, 0.5))
                stack = stack[1:]
            else:
                # left cycle
                res.append(cycle_point_numbers(a, b, 1))
                last = stack[-1]
                stack = stack[:-3]
                stack.append(last)
    else:
        while len(stack) > 1:
            res.append(cycle_point_numbers(stack[0], stack[1], 0.5))
            stack = stack[1:]

    return res


def pick_cycles_as_df(df, name):
    res = pick_cycles(df, name)
    df = pandas.DataFrame(res, columns=['Range', 'Count', 'Mean', 'Min', 'Max'])
    return df


def pick_cycles_point_number_as_df(df, name):
    res = pick_cycles_point_numbers(df, name)
    df = pandas.DataFrame(res, columns=['Min_point', 'Max_point', "Count"])
    return df


def calc_cycles_parameters_by_numbers(df_input, name, df_cycles, traces=[], dt=None):
    """

    :param df_input: dataFrame with input signals
    :param name: input signal name
    :param df_cycles: dataFrame with cycles point numbers
    :param traces: array with traces names
    :param dt: max time interval
    :return: dataFrame with cycles parameters = 'Range', 'Mean', 'Min', 'Max', 'Count', ['traces_names']
    """
    rows, _ = df_cycles.shape
    cols = df_input.columns
    data = []
    for i in range(rows):
        ind1 = df_cycles['Min_point'][i]
        ind2 = df_cycles['Max_point'][i]
        # print('ind1={}, ind2={}'.format(ind1, ind2))
        c = df_cycles['Count'][i]
        t1 = df_input[cols[0]][ind1]
        t2 = df_input[cols[0]][ind2]
        x1 = df_input[name][ind1]
        x2 = df_input[name][ind2]
        res = cycle_parameters(x1, x2, c)
        for trace in traces:
            min_ind = min(ind1, ind2)
            max_ind = max(ind1, ind2)
            # print('trace={}\n'.format(df_input[trace][min_ind:max_ind + 1]))
            if (dt is None) or (abs(t2 - t1) < dt):
                mean_trace = numpy.mean(df_input[trace][min_ind:max_ind + 1])
            else:
                mean_trace = numpy.inf
            res.append(mean_trace)
        data.append(res)
    df = pandas.DataFrame(data, columns=['Range', 'Count', 'Mean', 'Min', 'Max'] + traces)
    return df


def cumulative_frequency_extended(classes, data, names, calc_log=False):
    """
    Cumulative distribution function and traces
    :param classes: array of strings with classes intervals
    :param data: cycles counts
    :param traces: additional parameters values
    :param names: columns names
    :return: dataFrame index=classes points from min to max, columns = cumulative distribution function, traces
    """
    a0, _ = classes[0].split('-')
    xx = [a0]
    cdf = [0.0]
    for pnt in classes:
        _, a1 = pnt.split('-')
        xx.append(a1)
    for dat in data[0]:
        cdf.append(cdf[-1] + dat)
    if calc_log:
        cdf = numpy.log10(cdf)
    print('xx={}'.format(xx))
    print('cdf={}'.format(cdf))
    df = pandas.DataFrame(list(zip(xx, cdf)), columns=['Range', 'CDF'])
    print('zip={}'.format(list(zip(xx, cdf))))
    for i in range(1, len(names)):
        df[names[i]] = numpy.concatenate([[data[i][0]], data[i]])
    return df


def cumulative_frequency(classes, data, names, calc_log=False):
    """
    Cumulative distribution function and traces
    :param classes: array of strings with classes intervals
    :param data: cycles counts
    :param names: columns names
    :param calc_log: calc in log10 scale
    :return: dataFrame index=classes points from min to max, columns = cumulative distribution function, traces
    """
    xx = []
    cdf = [sum(data[0][:])]
    for pnt in classes:
        _, a1 = pnt.split('-')
        xx.append(a1)
    for i in range(len(data[0])):
        if sum(data[0][i + 1:]) > 0:
            cdf.append(sum(data[0][i + 1:]))
        else:
            break
    print(cdf)
    if calc_log:
        cdf = numpy.log10(cdf)
    df = pandas.DataFrame(list(zip(xx, cdf)), columns=['Range', 'CDF'])
    for i in range(1, len(names)):
        df[names[i]] = data[i][:len(cdf)]
    return df
