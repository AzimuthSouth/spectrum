import unittest
import numpy
import pandas

from staff import prepare


class TestPrepareData(unittest.TestCase):
    def test_smoothing_symm(self):
        df1 = pandas.DataFrame(list(zip(range(100), numpy.ones(100), 2 * numpy.ones(100))), columns=['x', 'y', 'z'])
        res1 = prepare.smoothing_symm(df1, 'y', 4, 1)
        ans1 = pandas.DataFrame(list(zip(range(1, 98), numpy.ones(97), 2 * numpy.ones(97))), columns=['x', 'y', 'z'])
        eps1 = numpy.sum((res1 - ans1).to_numpy() ** 2)

        df2 = pandas.DataFrame(list(zip(range(101), numpy.ones(101), 2 * numpy.ones(101))), columns=['x', 'y', 'z'])
        res2 = prepare.smoothing_symm(df2, 'y', 4, 1)
        ans2 = pandas.DataFrame(list(zip(range(1, 99), numpy.ones(98), 2 * numpy.ones(98))), columns=['x', 'y', 'z'])
        eps2 = numpy.sum((res2 - ans2).to_numpy() ** 2)
        self.assertEqual(eps1 < 1.0e-6 and eps2 < 1.0e-6, True)

    def test_set_smoothing_symm(self):
        df1 = pandas.DataFrame(list(zip(range(100), numpy.ones(100), 2 * numpy.ones(100))), columns=['x', 'y', 'z'])
        res1 = prepare.set_smoothing_symm(df1, ['y', 'z'], 4, 1)
        ans1 = pandas.DataFrame(
            list(zip(range(1, 98), numpy.ones(97), 2 * numpy.ones(97), numpy.ones(97), 2 * numpy.ones(97))),
            columns=['x', 'y', 'z', 'y_smooth', 'z_smooth'])
        eps1 = numpy.sum((res1 - ans1).to_numpy() ** 2)
        self.assertEqual(eps1 < 1.0e-6, True)

    def test_hann_coefficient_zero(self):
        self.assertEqual(prepare.hann_coefficient(0, 0), -1)

    def test_hann_coefficient(self):
        r = [0, 50, 100]
        n = [100] * 3
        res = numpy.array([prepare.hann_coefficient(r[i], n[i]) for i in range(len(r))])
        ans = numpy.array([0.0, 1.0, 0.0])
        eps = numpy.sum((res - ans) ** 2)
        self.assertEqual(eps < 1.0e-6, True)


if __name__ == '__main__':
    unittest.main()
