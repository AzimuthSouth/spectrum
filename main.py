# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import pandas as pd
from staff import pipeline
from research import examples
from staff import schematisation
import numpy
import pandas
import matplotlib.pyplot as plt


def test():
    t = numpy.arange(0.0, 1.0, 0.01)
    f1 = 3.5
    f2 = 11.0
    y = numpy.cos(2 * numpy.pi * f1 * t) + 2 * numpy.cos(2 * numpy.pi * f2 * t)
    data = numpy.array(list(zip(t, y)))
    df = pandas.DataFrame(data, columns=['t', 'x'])
    res = schematisation.max_frequency(df, 'x', 3)
    print(res)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    test()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
