# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy
from staff import schematisation
from staff import draw
from research import examples


def test():
    rnd = 10 * numpy.random.rand(100)
    data = []
    for i in range(len(rnd)):
        data.append([i, rnd[i] + 100 * numpy.cos(2 * numpy.pi * 4 * i)])
    data = numpy.array(data)
    sch = schematisation.merge(data, 1, 2.0)

    p = draw.Painter()
    p.draw_uncommon([data[:, 0], sch[:, 0]], [data[:, 1], sch[:, 1]], "time", "input", ["raw", "schem"])
    p.draw_classes(sch, schematisation.set_classes(sch, 14), "time", "input")


def test1():
    data = numpy.array([[0.0, 0.0], [1.0, 3.0], [2.0, 2.0], [3.0, 3.2], [4.0, 0.0]])
    c = schematisation.extreme_count(data[:, 1])
    print(c)
    print(schematisation.merge(data, 1, 1.5))
    c = schematisation.extreme_count(schematisation.merge(data, 1, 1.5)[:, 1])
    print(c)


def test2():
    time = numpy.arange(101)
    data = numpy.array([[i, i + 1] for i in time])
    print(data)
    seg = schematisation.set_n_segments(data, 10)
    print(seg)
    fmax = schematisation.max_frequency(data, 10, 1.0)
    print(fmax)


def test3():
    data1 = numpy.array([[0.0, 500.0], [1.0, -200.0], [2.0, -100.0], [3.0, -200.0], [4.0, 200.0],
                        [5.0, -50.0], [6.0, 500.0]])
    data2 = numpy.array([[0.0, 100.0], [1.0, -100.0], [2.0, -50.0], [3.0, -100.0], [4.0, 150.0],
                        [5.0, 100.0], [6.0, 150.0], [7.0, 100]])
    data3 = numpy.array([[0.0, 150.0], [1.0, 100.0], [2.0, 150.0], [3.0, -100.0], [4.0, -50.0],
                        [5.0, -100.0], [6.0, 150.0]])

    print(schematisation.count_cycles(data2))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    test3()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
