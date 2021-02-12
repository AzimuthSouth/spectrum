from collections import defaultdict
import math
import numpy


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
            if len(buf) > 1:
                res.append(buf[-1])
                res.append(x[i])
            else:
                res.append(x[i])
            buf = []

        if i == (len(x) - 1) and len(buf) > 1:
            res.append(x[i])

    return numpy.array(res)


def is_extreme(x, delta):
    if (abs(x[0] - x[1]) >= delta) or (abs(x[2] - x[1]) >= delta):
        if (x[0] < x[1]) and (x[1] > x[2]):
            return True
        if (x[0] > x[1]) and (x[1] < x[2]):
            return True
    return False


def extreme_count(x, delta=0):
    # first and last points are not extremes
    res = 0
    for i in range(1, len(x) - 1):
        if is_extreme(x[i - 1:i + 2], delta):
            res += 1
    return res


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


def max_frequency(x, n, delta):
    dt = (numpy.max(x[:, 0]) - numpy.min(x[:, 0])) / n
    seg = set_n_segments(x, dt)
    f = []
    for i in range(len(seg)):
        dti = seg[i][-2, 0] - seg[i][1, 0]
        f.append(extreme_count(seg[i][:, 1], delta) / (2 * dti))
    return numpy.max(f)


def mean_count(x):
    res = 0
    mn = numpy.mean(x[:, 1])
    for i in range(1, len(x)):
        if (x[i - 1, 1] <= mn) and (x[i, 1] > mn):
            res += 1
    return res


def input_stats(data, m):
    mn = numpy.mean(data[:, 1])
    s2 = numpy.var(data[:, 1])
    st = numpy.std(data[:, 1])
    kp = mean_count(data) / extreme_count(data, class_width(data, m))
    return [mn, s2, st, kp]


def pick_extremes(data, delta):
    x = data[:, 1]
    res = [data[0]]
    for i in range(1, len(data) - 1):
        if is_extreme(x[i - 1:i + 2], delta):
            res.append(data[i])
    res.append(data[-1])
    return res


def range_mean(x1, x2, c):
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
                res.append(range_mean(a, b, 0.5))
                stack = stack[1:]
            else:
                # left cycle
                res.append(range_mean(a, b, 1))
                last = stack[-1]
                stack = stack[:-3]
                stack.append(last)
    else:
        while len(stack) > 1:
            res.append(range_mean(stack[0], stack[1], 0.5))
            stack = stack[1:]

    return res


def count_cycles(data, ndigits=None, n_seg=None, seg_size=None):

    counts = defaultdict(float)
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
