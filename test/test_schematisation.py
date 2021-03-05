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
        self.x4 = numpy.array([[0.0, 3.0], [1.0, 5.0], [2.0, 3.0], [3.0, 5.0], [4.0, 3.0], [5.0, 5.0],
                               [9.0, -5.0], [10.0, -3.0], [11.0, -5.0], [12.0, -3.0], [13.0, -5.0], [14.0, -3.0]])
        self.x5 = numpy.array([[5.0, 5.0], [9.0, -5.0]])
        self.x6 = numpy.array([[0.0, -3.0], [1.0, -5.0], [2.0, -3.0], [3.0, -5.0], [4.0, -3.0], [5.0, -5.0],
                               [9.0, 5.0], [10.0, 3.0], [11.0, 5.0], [12.0, 3.0], [13.0, 5.0], [14.0, 3.0]])
        self.x7 = numpy.array([[5.0, -5.0], [9.0, 5.0]])
        a = 9.980267e-1
        self.x8 = numpy.array([[0.0, 0.0, 1.0], [0.06, a, 1.0], [0.19, -a, 1.0], [0.31, a, 1.0], [0.44, -a, 1.0],
                               [0.56, a, 1.0], [0.69, -a, 1.0], [0.81, a, 1.0], [0.94, -a, 1.0], [1.0, 0.0, 1.0]])
        self.x9 = [0.0 if i % 2 == 0 else 1.0 for i in range(10)]
        self.x10 = [-2.0, 1.0, -3.0, 5.0, -1.0, 3.0, -4.0, 4.0, -2.0]
        self.x11 = [[3.0, 0.5, -0.5, -2.0, 1.0], [4.0, 0.5, -1.0, -3.0, 1.0], [4.0, 1, 1.0, -1.0, 3.0],
                    [8.0, 0.5, 1.0, -3.0, 5.0], [9.0, 0.5, 0.5, -4.0, 5.0], [8.0, 0.5, 0.0, -4.0, 4.0],
                    [6.0, 0.5, 1.0, -2.0, 4.0]]
        self.x11a = [[0, 1, 0.5], [2, 1, 0.5], [4, 5, 1.0], [2, 3, 0.5], [6, 3, 0.5], [6, 7, 0.5], [8, 7, 0.5]]
        self.x12 = [[0.0, 0.0], [1.0, 2.0], [3.0, 1.0], [4.0, 3.0], [5.0, 2.0], [6.0, 8.0], [7.0, 6.0], [8.0, 8.0]]
        self.x13 = [[0.0, 0.0], [6.0, 8.0], [7.0, 6.0], [8.0, 8.0]]
        self.x14 = [[0.0, 0.0], [1.0, 1.0], [2.0, 0.5], [3.0, 1.5], [4.0, 0.5], [5.0, 1.0], [6.0, 0.0], [7.0, 0.5],
                    [8.0, -3.0], [9.0, -1.0]]
        self.x15 = [[7.0, 0.5], [8.0, -3.0], [9.0, -1.0]]
        self.x16 = [[0.0, 0.0], [1.0, 2.0], [3.0, 1.0], [4.0, 3.0], [5.0, 2.0], [6.0, 3.0], [7.0, 1.0], [8.0, 2.0],
                    [9.0, 0.0]]
        self.x17 = [[0.0, 0.0], [4.0, 3.0], [9.0, 0.0]]
        self.x18 = [[0.0, 0.0], [1.0, 5.0], [2.0, -1.0], [3.0, 6.0], [4.0, 3.0], [5.0, 4.0], [6.0, 3.0], [7.0, 4.0],
                    [8.0, 3.0], [9.0, 4.0]]
        self.x19 = [[0.0, 0.0], [1.0, 5.0], [2.0, -1.0], [3.0, 6.0], [4.0, 3.0]]
        self.x20 = [[3.0, 0.5, -0.5, -2.0, 1.0, 0.5],
                    [4.0, 0.5, -1.0, -3.0, 1.0, 1.5],
                    [4.0, 1.0, 1.0, -1.0, 3.0, 4.5],
                    [8.0, 0.5, 1.0, -3.0, 5.0, 2.5],
                    [9.0, 0.5, 0.5, -4.0, 5.0, 4.5],
                    [8.0, 0.5, 0.0, -4.0, 4.0, 6.5],
                    [6.0, 0.5, 1.0, -2.0, 4.0, 7.5]]
        self.x21 = [[0.0, 0.0], [4.0, 0.0]]
        self.x22 = [[0.0, 0.0], [3.8333333, 0.0]]
        self.x23 = [[0.0, 0.0], [1.0, 0.0]]

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

    def test_extreme_merge(self):
        self.setdata()
        df = pd.DataFrame(self.x4, columns=['t', 'x'])
        res = schematisation.merge_extremes(df, 'x', 3.0)
        ans = pd.DataFrame(self.x5, columns=['t', 'x'])
        eps = [numpy.linalg.norm(res - ans)]
        df = pd.DataFrame(self.x6, columns=['t', 'x'])
        res = schematisation.merge_extremes(df, 'x', 3.0)
        ans = pd.DataFrame(self.x7, columns=['t', 'x'])
        eps.append(numpy.linalg.norm(res - ans))
        df = pd.DataFrame(self.x12, columns=['t', 'x'])
        res = schematisation.merge_extremes(df, 'x', 2.0)
        ans = pd.DataFrame(self.x13, columns=['t', 'x'])
        eps.append(numpy.linalg.norm(res - ans))
        df = pd.DataFrame(self.x14, columns=['t', 'x'])
        res = schematisation.merge_extremes(df, 'x', 2.0)
        ans = pd.DataFrame(self.x15, columns=['t', 'x'])
        eps.append(numpy.linalg.norm(res - ans))
        df = pd.DataFrame(self.x16, columns=['t', 'x'])
        res = schematisation.merge_extremes(df, 'x', 2.0)
        ans = pd.DataFrame(self.x17, columns=['t', 'x'])
        eps.append(numpy.linalg.norm(res - ans))
        df = pd.DataFrame(self.x18, columns=['t', 'x'])
        res = schematisation.merge_extremes(df, 'x', 2.0)
        ans = pd.DataFrame(self.x19, columns=['t', 'x'])
        eps.append(numpy.linalg.norm(res - ans))

        self.assertEqual(numpy.linalg.norm(eps) < 1.0e-6, True)

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

    def test_mean_count(self):
        t = numpy.arange(0.0, 1.01, 0.01)
        f = 4
        y = numpy.sin(2 * numpy.pi * f * t)
        res = schematisation.mean_count(y)
        self.assertEqual(abs(res - (2 * f - 1)) < 1.0e-4, True)
        pass

    def test_pick_extremes(self):
        self.setdata()
        df = pd.DataFrame(self.x4, columns=['t', 'x'])
        res = schematisation.pick_extremes(df, 'x')
        ans = pd.DataFrame(self.x4, columns=['t', 'x'])
        eps = [numpy.linalg.norm(res - ans)]
        t = numpy.arange(0.0, 1.01, 0.01)
        f = 4
        y = numpy.sin(2 * numpy.pi * f * t)
        trace = numpy.ones(len(t))
        data = numpy.array(list(zip(t, y, trace)))
        df = pd.DataFrame(data, columns=['t', 'x', 'tg'])
        res = schematisation.pick_extremes(df, 'x')
        eps.append(numpy.linalg.norm(res - self.x8))
        self.assertEqual(numpy.linalg.norm(eps) < 1.0e-6, True)

    def test_repetition_rate(self):
        self.setdata()
        res = schematisation.repetition_rate(self.x9, count=2)
        ans = [(0.5, 5.0), (1.0, 5.0)]
        self.assertEqual(res, ans)

    def test_repetition_rate_1(self):
        data = numpy.arange(0, 1.01, 0.01)
        res = schematisation.repetition_rate(data, count=2)
        ans = [(0.5, 50.0), (1.0, 51.0)]
        self.assertEqual(res, ans)

    def test_correlation_table(self):
        self.setdata()
        df = pd.DataFrame(self.x11, columns=['Range', 'Count', 'Mean', 'Min', 'Max'])
        res = schematisation.correlation_table(df, 'Min', 'Max', -4.0, 5.0, 3).to_numpy()
        ans = [[0.0, 1.0, 2.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]
        eps = [numpy.linalg.norm(res - ans)]
        res = schematisation.correlation_table(df, 'Range', 'Mean', -1.0, 9.0, 2).to_numpy()
        ans = [[0.5, 0], [3.5, 0]]
        eps.append(numpy.linalg.norm(res - ans))
        self.assertEqual(numpy.linalg.norm(eps) < 1.0e-6, True)

    def test_pick_cycles(self):
        self.setdata()
        t = range(len(self.x10))
        data = list(zip(t, self.x10))
        df = pd.DataFrame(data, columns=['t', 'x'])
        res = numpy.array(schematisation.pick_cycles(df, 'x'))
        eps = numpy.linalg.norm(res - self.x11)
        self.assertEqual(numpy.linalg.norm(eps) < 1.0e-6, True)

    def test_pick_cycles_numbers(self):
        self.setdata()
        t = range(len(self.x10))
        data = list(zip(t, self.x10))
        df = pd.DataFrame(data, columns=['t', 'x'])
        res = numpy.array(schematisation.pick_cycles_point_numbers(df, 'x'))
        eps = numpy.linalg.norm(res - self.x11a)
        self.assertEqual(numpy.linalg.norm(eps) < 1.0e-6, True)

    def test_pick_cycles_parameters_by_number(self):
        self.setdata()
        t = range(len(self.x10))
        data = list(zip(t, self.x10, t))
        df = pd.DataFrame(data, columns=['t', 'x', 'trace'])
        cycles_number = schematisation.pick_cycles_point_number_as_df(df, 'x')
        res = schematisation.calc_cycles_parameters_by_numbers(df, 'x', cycles_number, ['trace'], 3.5)
        eps = numpy.linalg.norm(res.to_numpy() - self.x20)
        self.assertEqual(numpy.linalg.norm(eps) < 1.0e-6, True)

    def test_correlation_table_with_traces(self):
        self.setdata()
        t = range(len(self.x10))
        tr = [1] * len(t)
        data = list(zip(t, self.x10, t, tr))
        df = pd.DataFrame(data, columns=['t', 'x', 'trace', 'one'])
        cycles_number = schematisation.pick_cycles_point_number_as_df(df, 'x')
        res = schematisation.calc_cycles_parameters_by_numbers(df, 'x', cycles_number, ['trace', 'one'], 1.5)
        tbl, traces = schematisation.correlation_table_with_traces(res, 'Max', 'Min', ['trace', 'one'], count=2)
        eps = [numpy.linalg.norm(tbl.to_numpy() - self.x21), numpy.linalg.norm(traces[0].to_numpy() - self.x22),
               numpy.linalg.norm(traces[1].to_numpy() - self.x23)]
        self.assertEqual(numpy.linalg.norm(eps) < 1.0e-6, True)
