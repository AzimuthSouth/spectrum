import numpy


def nominal_parameters(x):
    x_m = numpy.mean(x)
    x_std = numpy.std(x)
    return [x_m, x_std]


def design_parameters(x):
    z_a = 2.32
    nominal = nominal_parameters(x)
    n = len(x)
    x_m = nominal[0] + z_a * nominal[1] / (n ** 0.5)
    x_std = (2 * n) ** 0.5 / ((2 * n - 3) ** 0.5 - z_a) * nominal[1]
    return [x_m, x_std]
