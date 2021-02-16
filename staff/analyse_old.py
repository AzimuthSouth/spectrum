import numpy
from scipy import signal

def frequencies(n, dt):
    f = numpy.zeros(n)
    for i in range(n):
        f[i] = i / n / dt
    return f[:int(n / 2)]


def spectrum_density(x, k):
    n = len(x)
    xx = numpy.fft.fft(x) / n
    g_xx = k * 2 * (abs(xx) ** 2)
    return g_xx[:int(n / 2)]


def n_coefficient(z):
    if z.real >= 0:
        return 0
    if (z.real < 0) and (z.imag >= 0):
        return 1
    return -1


def cross_spectrum(x, y):
    if len(x) != len(y):
        return [-1, "Streams x and y have different sizes"]
    n = len(x)
    xx = numpy.fft.fft(x) / n
    xx_ = numpy.zeros(n, dtype=complex)
    for i in range(n):
        xx_[i] = xx[i].conjugate()
    yy = numpy.fft.fft(y) / n
    g_xy = xx_ * yy
    g_xy[1:] *= 2.0
    return g_xy[:int(n / 2)]


def cross_spectrum_mod_fas(g_xy):
    n = len(g_xy)
    mod = numpy.zeros(n)
    fas = numpy.zeros(n)
    for i in range(n):
        mod[i] = abs(g_xy[i])
        if g_xy[i].real == 0:
            if g_xy[i].imag >= 0:
                fas[i] = numpy.pi / 2
            else:
                fas[i] = -numpy.pi / 2
        else:
            fas[i] = numpy.arctan(g_xy[i].imag / g_xy[i].real) + n_coefficient(g_xy[i]) * numpy.pi
    return [mod, fas]


def coherent_coefficient(g_xy, g_xx, g_yy):
    n = len(g_xy)
    if len(g_xx) != n:
        return [-1, "g_xx and g_xy have different sizes"]
    if len(g_yy) != n:
        return [-1, "g_yy and g_xy have different sizes"]
    gama2_xy = numpy.zeros(n, dtype=float)
    for i in range(n):
        a = abs(g_xy[i]) ** 2
        b = (abs(g_xx[i]) * abs(g_yy[i]))
        gama2_xy[i] = a / b
        if i in range(110, 130):
            print("g_xy**2 ={} g_xx*g_yy ={} gama ={}".format(a, b, a / b))

    return gama2_xy


def coherent_function(x, y, n):
    m = int(len(x) / n)
    gama = []
    for i in range(m):
        g_xy = cross_spectrum(x[i * n:i * (n + 1)], y[i * n:i * (n + 1)])
        g_xx = cross_spectrum(x[i * n:i * (n + 1)], x[i * n:i * (n + 1)])
        g_yy = cross_spectrum(y[i * n:i * (n + 1)], y[i * n:i * (n + 1)])
        gama.append(coherent_coefficient(g_xy, g_xx, g_yy))
    print(gama)

