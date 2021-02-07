import numpy
from staff import prepare
from staff import draw
from staff import analyse

from test import test_prepare


def test():
    data = 100 * numpy.random.rand(50)
    res = prepare.smoothing(data, 4)
    wres = prepare.correction(res, 2)

    p = draw.Painter()
    p.draw(range(len(res)), [data, res, wres], ["raw", "smooth", "hann"], ["time", "amplitude"])


def test1():
    t = numpy.arange(0, 10.0, 0.01)
    f0 = 4.0
    x = 3.0 * numpy.cos(2 * numpy.pi * f0 * t)
    res = prepare.smoothing(x, 4)
    wres = prepare.correction(res, 2)

    p = draw.Painter()
    p.draw(range(len(res)), [x, res, wres], ["raw", "smooth", "hann"], ["time", "amplitude"])


def test2():
    tp = test_prepare.TestPrepareData()
    tp.draw_weight()


def test3():
    fs = 1e3
    n = 1e4
    t = numpy.arange(n) / fs
    f0 = 4.0
    f1 = 25.0
    x = 3.0 * numpy.cos(2 * numpy.pi * f0 * t) + 5.0 * numpy.cos(2 * numpy.pi * f1 * t)
    # x = numpy.random.standard_normal(size=len(t))
    f, g = analyse.spectrum_density(x, fs, 150)
    _, g_xx = analyse.cross_spectrum(x, x, fs, 150)
    p = draw.Painter()
    p.draw(f, [g], ["spectrum density"], ["frequencies", "spectrum density"])

    p.draw_n(f, [g, abs(g_xx) - g], ["spectrum density", "cross_xx - density"], ["frequencies", ""], 1)


def test4():
    fs = 1e3
    n = 1e4
    t = numpy.arange(n) / fs
    f0 = 12.0
    f1 = 4.0
    x = 3.0 * numpy.cos(2 * numpy.pi * f0 * t)
    y = 4.0 * numpy.cos(2 * numpy.pi * f1 * t) + numpy.cos(2 * numpy.pi * f0 * t + numpy.pi / 4)
    f, g = analyse.cross_spectrum(x, y, fs, len(t))
    [mod, fas] = analyse.cross_spectrum_mod_fas(g)
    p = draw.Painter()
    p.draw_n(f, [mod, fas], ["Module", "Phase"], ["frequencies"])
    print(f[120])
    print(mod[120])
    print(fas[120])


def test5():
    fs = 1e3
    n = 1e4
    t = numpy.arange(n) / fs
    f0 = 12.0
    f1 = 4.0
    x = 3.0 * numpy.cos(2 * numpy.pi * f0 * t) + 20 * numpy.random.standard_normal(size=len(t))
    y = 4.0 * numpy.sin(2 * numpy.pi * f1 * t) + 2 * numpy.cos(2 * numpy.pi * f0 * t)
    f, g = analyse.coherent_function(x, y, fs, 1024)
    _, g_xy = analyse.cross_spectrum(x, y, fs, 1024)
    _, g_xx = analyse.cross_spectrum(x, x, fs, 1024)
    _, g_yy = analyse.cross_spectrum(y, y, fs, 1024)
    p = draw.Painter()
    p.draw_n(t, [x, y], ["x", "y"],
             ["time", "input"])
    p.draw_n(f, [abs(g_xy), abs(g_xx), abs(g_yy), g], ["cross_xy", "cross_xx", "cross_yy", "coherent_coef"],
             ["frequencies"], 1)
    p.draw(f, [g], ["coherent coefficient"], ["frequencies", "coherent_coef"], 1)
