import numpy
from staff import prepare
from staff import draw
from staff import analyse
from staff import schematisation
from staff import pipeline
from test import test_prepare
import pandas as pd


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


def test6():
    rnd = 10 * numpy.random.rand(100)
    data = []
    for i in range(len(rnd)):
        data.append([i, rnd[i] + 100 * numpy.cos(2 * numpy.pi * 4 * i)])
    data = numpy.array(data)
    sch = schematisation.merge(data, 1, 2.0)

    p = draw.Painter()
    p.draw_uncommon([data[:, 0], sch[:, 0]], [data[:, 1], sch[:, 1]], "time", "input", ["raw", "schem"])
    p.draw_classes(sch, schematisation.set_classes(sch, 14), "time", "input")


def test7():
    data = numpy.array([[0.0, 0.0], [1.0, 3.0], [2.0, 2.0], [3.0, 3.2], [4.0, 0.0]])
    c = schematisation.extreme_count(data[:, 1])
    print(c)
    print(schematisation.merge(data, 1, 1.5))
    c = schematisation.extreme_count(schematisation.merge(data, 1, 1.5)[:, 1])
    print(c)


def test8():
    time = numpy.arange(101)
    data = numpy.array([[i, i + 1] for i in time])
    print(data)
    seg = schematisation.set_n_segments(data, 10)
    print(seg)
    fmax = schematisation.max_frequency(data, 10, 1.0)
    print(fmax)


def test9():
    data1 = numpy.array([[0.0, 500.0], [1.0, -200.0], [2.0, -100.0], [3.0, -200.0], [4.0, 200.0],
                        [5.0, -50.0], [6.0, 500.0]])
    data2 = numpy.array([[0.0, 100.0], [1.0, -100.0], [2.0, -50.0], [3.0, -100.0], [4.0, 150.0],
                        [5.0, 100.0], [6.0, 150.0], [7.0, 100]])
    data3 = numpy.array([[0.0, 150.0], [1.0, 100.0], [2.0, 150.0], [3.0, -100.0], [4.0, -50.0],
                        [5.0, -100.0], [6.0, 150.0]])
    # stream from GOST 25.101-83
    data4 = numpy.array([[0.0, 0.0], [1.0, 10.0], [2.0, -10.0], [3.0, 40.0], [4.0, 30.0], [5.0, 40.0],
                         [6.0, 10.0], [7.0, 20.0], [8.0, -20.0], [9.0, -10.0], [10.0, -30.0],
                         [11.0, 30.0], [12.0, 20.0], [13.0, 70.0], [14.0, 0.0], [15.0, 10.0],
                         [16.0, -40.0], [17.0, 0.0], [18.0, -20.0], [19.0, 50.0], [20.0, -10.0],
                         [21.0, 10.0], [22.0, 0.0], [23.0, 30.0], [24.0, 0.0]])

    print(schematisation.count_cycles(data4))


def test10():
    # stream from GOST 25.101-83
    data4 = numpy.array([[0.0, 0.0], [1.0, 10.0], [2.0, -10.0], [3.0, 40.0], [4.0, 30.0], [5.0, 40.0],
                         [6.0, 10.0], [7.0, 20.0], [8.0, -20.0], [9.0, -10.0], [10.0, -30.0],
                         [11.0, 30.0], [12.0, 20.0], [13.0, 70.0], [14.0, 0.0], [15.0, 10.0],
                         [16.0, -40.0], [17.0, 0.0], [18.0, -20.0], [19.0, 50.0], [20.0, -10.0],
                         [21.0, 10.0], [22.0, 0.0], [23.0, 30.0], [24.0, 0.0]])
    print(len(data4))
    print(schematisation.mean_count(data4[:, 1]))
    # print(schematisation.extremes_method(data4, 10))


def test11():
    df = pd.DataFrame(list(zip(range(100), numpy.ones(100), 2 * numpy.ones(100))), columns=['x', 'y', 'z'])
    print(pipeline.signal_processing(df, 'y', 1.0, 1))


def test12():
    # df = pandas.DataFrame(res, columns=['Range', 'Count', 'Mean', 'Min', 'Max'])
    arr = [[2.0, 1.0, 5.0, 4.0, 6.0],
           [5.0, 1.0, 6.5, 4.0, 9.0],
           [1.0, 1.0, 8.5, 8.0, 9.0],
           [1.0, 1.0, 8.5, 8.0, 9.0],
           [3.0, 1.0, 7.5, 6.0, 9.0],
           [1.0, 1.0, 6.5, 6.0, 7.0],
           [4.0, 1.0, 5.0, 3.0, 7.0],
           [1.0, 1.0, 7.5, 3.0, 4.0],
           [2.0, 1.0, 3.0, 2.0, 4.0],
           [6.0, 1.0, 5.0, 2.0, 8.0],
           [1.0, 1.0, 7.5, 7.0, 8.0],
           [5.0, 1.0, 9.5, 7.0, 12.0],
           [7.0, 1.0, 8.5, 5.0, 12.0],
           [1.0, 1.0, 5.5, 5.0, 6.0],
           [5.0, 1.0, 3.5, 1.0, 6.0],
           [4.0, 1.0, 3.0, 1.0, 5.0],
           [2.0, 1.0, 4.0, 3.0, 5.0],
           [7.0, 1.0, 6.5, 3.0, 10.0],
           [6.0, 1.0, 7.0, 4.0, 10.0],
           [2.0, 1.0, 5.0, 4.0, 6.0],
           [1.0, 1.0, 5.5, 5.0, 6.0],
           [3.0, 1.0, 6.5, 5.0, 8.0]]
    df = pd.DataFrame(arr, columns=['Range', 'Count', 'Mean', 'Min', 'Max'])
    print(schematisation.correlation_table(df, 'Range', 'Mean', 12))
