# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy
from staff import schematisation


def test():
    x = numpy.array([[0.0, 0.0],
                     [1.0, 3.0],
                     [2.0, 2.0],
                     [3.0, 3.0],
                     [4.0, 0.0]])
    res = schematisation.merge(x, 1, 1.5)
    print(res)
    ans = numpy.array([[0.0, 0.0],
                       [1.0, 3.0],
                       [3.0, 3.0],
                       [4.0, 0.0]])
    print(ans)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    test()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
