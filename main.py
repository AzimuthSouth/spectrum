# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy
from staff import schematisation
from staff import draw
from research import examples


def test():
    rnd = 100 * numpy.random.rand(50)
    data = []
    for i in range(len(rnd)):
        data.append([i, rnd[i]])
    sch = schematisation.merge(data, 1, 10.0)
    data = []
    for i in range(len(sch)):
        data.append(sch[i][1])
    p = draw.Painter()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # test()
    examples.test5()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
