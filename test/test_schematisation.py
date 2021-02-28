import unittest
import numpy
import pandas as pd
from staff import schematisation


class TestSchematisationData(unittest.TestCase):
    def setdata(self):
        self.x1 = numpy.array([[0.0, 0.0], [1.0, 0.5], [2.0, 1.0], [3.0, 1.5], [4.0, 2.0]])
        self.x2 = numpy.array([[0.0, 0.0], [1.0, 3.0], [2.0, 2.0], [3.0, 3.0], [4.0, 0.0]])
        self.x3 = numpy.array([[0.0, 6.0], [1.0, 12.0], [2.0, 10.0], [3.0, 11.0], [4.0, 8.0],
                               [5.0, 6.0], [6.0, 5.5], [7.0, 6.0], [8.0, 5.5], [9.0, 5.0], [10.0, 4.5],
                               [11.0, 4.0], [12.0, 7.0], [13.0, 10.0], [14.0, 11.0], [15.0, 11.5], [16.0, 11.0]])

    def test_merge_trend(self):
        self.setdata()
        df = pd.DataFrame(self.x1, columns=['x', 'y'])
        res = schematisation.merge(df, 'y', 1.0)
        ans = pd.DataFrame([[0.0, 0.0], [4.0, 2.0]], columns=['x', 'y'])
        self.assertEqual(numpy.linalg.norm(res - ans) < 1.0e-7, True)

    def test_merge_local_extreme(self):
        self.setdata()
        df = pd.DataFrame(self.x2, columns=['x', 'y'])
        res = schematisation.merge(df, 'y', 1.5)
        ans = pd.DataFrame([[0.0, 0.0], [2.0, 2.6666666], [4.0, 0.0]], columns=['x', 'y'])
        self.assertEqual(numpy.linalg.norm(res - ans) < 1.0e-7, True)

    def test_merge_25(self):
        self.setdata()
        df = pd.DataFrame(self.x3, columns=['x', 'y'])
        res = schematisation.merge(df, 'y', 0.9)
        ans = pd.DataFrame([[0.0, 6.0], [1.0, 12.0], [2.0, 10.0], [3.0, 11.0], [4.0, 8.0], [8.0, 5.214286],
                            [12.0, 7.0], [13.0, 10.0], [16.0, 11.0]], columns=['x', 'y'])
        self.assertEqual(numpy.linalg.norm(res - ans) < 1.0e-6, True)

    def test_extreme_count(self):
        self.setdata()
        res = numpy.array([schematisation.extreme_count(self.x1[:, 1]), schematisation.extreme_count(self.x2[:, 1]),
                           schematisation.extreme_count(self.x3[:, 1])])
        ans = numpy.array([2, 5, 9])
        self.assertEqual(numpy.linalg.norm(res - ans) < 1.0e-7, True)

    def test_extreme_count_harmonic(self):
        t = numpy.arange(0.0, 1.01, 0.01)
        f = 4
        y = numpy.cos(2 * numpy.pi * f * t)
        self.assertEqual(schematisation.extreme_count(y), 9)

    def test_is_max(self):
        data = [[0.0, 1.0, 0.0], [0.0, 1.0, 1.0], [1.0, 0.0, 0.0]]
        res = [schematisation.is_max(i) for i in data]
        ans = [True, False, False]
        self.assertEqual(res, ans)

    def test_is_min(self):
        data = [[0.0, -1.0, 0.0], [0.0, 1.0, 1.0], [1.0, 0.0, 0.0]]
        res = [schematisation.is_min(i) for i in data]
        ans = [True, False, False]
        self.assertEqual(res, ans)

    def test_is_extreme(self):
        data = [[0.0, -1.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]
        res = [schematisation.is_extreme(i) for i in data]
        ans = [True, True, False]
        self.assertEqual(res, ans)

    def test_max_frequency_harmonic(self):
        t = numpy.arange(0.0, 1.0, 0.01)
        f = 4
        y = numpy.sin(2 * numpy.pi * f * t)
        data = numpy.array(list(zip(t, y)))
        df = pd.DataFrame(data, columns=['t', 'x'])
        res = schematisation.max_frequency(df, 'x', 1)
        self.assertEqual(abs(res - 3.977272) < 1.0e-4, True)

    def test_max_frequency_2harmonic(self):
        t = numpy.arange(0.0, 1.0, 0.01)
        f1 = 3.5
        f2 = 10.0
        y = numpy.cos(2 * numpy.pi * f1 * t) + 2 * numpy.cos(2 * numpy.pi * f2 * t)
        data = numpy.array(list(zip(t, y)))
        df = pd.DataFrame(data, columns=['t', 'x'])
        res = schematisation.max_frequency(df, 'x', 3)
        self.assertEqual(abs(res - f2) < 1.0e-5, True)
