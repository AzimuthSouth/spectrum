import numpy


def smoothing(data, k):
    res = []
    mn = numpy.mean(data)
    for i in range(len(data) - k + 1):
        xi = data[i:k + i]
        res.append(data[i] - numpy.mean(xi))
    # rest of data centering with mean of data
    tail = data[len(data) - k + 1: len(data)]
    tail -= mn
    res = numpy.concatenate((res, tail))
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


def correction_hann(data, par=None):
    res = []
    for i in range(len(data)):
        res.append(data[i] * hann_coefficient(i, len(data) - 1))
    return res


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
