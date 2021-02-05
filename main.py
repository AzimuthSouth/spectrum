# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
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
    t = numpy.arange(0, 10.0, 0.01)
    f0 = 4.0
    f1 = 25.0
    x = 3.0 * numpy.cos(2 * numpy.pi * f0 * t) + 5.0 * numpy.cos(2 * numpy.pi * f1 * t)
    f = analyse.frequencies(len(x), 0.01)
    g = analyse.spectrum_density(x, 1.0)
    g_xx = analyse.cross_spectrum(x, x)
    p = draw.Painter()
    p.draw(f, [g], ["spectrum density"], ["frequencies", "spectrum density"])

    p.draw_n(f, [g, abs(g_xx) - g], ["spectrum density", "cross_xx - density"], ["frequencies", ""], 1)


def test4():
    t = numpy.arange(0, 10.0, 0.01)
    f0 = 12.0
    f1 = 4.0
    x = 3.0 * numpy.cos(2 * numpy.pi * f0 * t)
    y = 4.0 * numpy.cos(2 * numpy.pi * f1 * t) + numpy.cos(2 * numpy.pi * f0 * t + numpy.pi / 4)
    f = analyse.frequencies(len(x), 0.01)
    g = analyse.cross_spectrum(x, y)
    [mod, fas] = analyse.cross_spectrum_mod_fas(g)
    p = draw.Painter()
    p.draw_n(f, [mod, fas], ["Module", "Phase"], ["frequencies"])


def test5():
    t = numpy.arange(0, 10.0, 0.005)
    f0 = 12.0
    f1 = 4.0
    x = 3.0 * numpy.cos(2 * numpy.pi * f0 * t)
    y = 4.0 * numpy.sin(2 * numpy.pi * f1 * t) + numpy.cos(2 * numpy.pi * f0 * t + numpy.pi / 4)
    f = analyse.frequencies(len(x), 0.005)
    g = analyse.coherent_function(x, y)
    g_xy = analyse.cross_spectrum(x, y)
    g_xx = analyse.cross_spectrum(x, x)
    g_yy = analyse.cross_spectrum(y, y)
    p = draw.Painter()
    p.draw_n(f, [abs(g_xy), abs(g_xx), abs(g_yy), g], ["cross_xy", "cross_xx", "cross_yy", "coherent_coef"],
             ["frequencies"], 1)
    p.draw(f, [g], ["coherent coefficient"], ["frequencies", "coherent_coef"], 1)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    test5()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
