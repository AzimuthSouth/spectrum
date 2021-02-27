import numpy
from scipy import signal

def frequencies(n, dt):
    f = numpy.zeros(n)
    for i in range(n):
        f[i] = i / n / dt
    return numpy.array(f[:int(n / 2) + 1])


def spectrum_density(x, fs, nps):
    f, g_xx = signal.csd(x, x, fs, window="boxcar", nperseg=nps)
    return [f, g_xx]


def n_coefficient(z):
    if z.real >= 0:
        return 0
    if (z.real < 0) and (z.imag >= 0):
        return 1
    return -1


def cross_spectrum(x, y, fs, nps):
    f, g_xy = signal.csd(x, y, fs, window="boxcar", nperseg=nps)
    return [f, g_xy]


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


def coherent_function(x, y, fs, nps):
    f, c_xy = signal.coherence(x, y, fs, nperseg=nps)
    return[f, c_xy]

