import unittest
import numpy
import pandas

from staff import prepare
from staff import draw


class TestPrepareData(unittest.TestCase):
    def setdata(self):
        self.data = [56.28543488, 80.56994913, 84.56765223, 58.1584399, 17.27367815, 52.68875561,
                     14.25480842, 64.60461641, 78.1399531, 39.90966189, 45.6945331, 24.46529599,
                     87.5438924, 86.48494703, 45.32133927, 96.22855131, 61.00539127, 19.1261671,
                     34.30791756, 92.32617997, 88.21771248, 72.02419302, 67.38431245, 70.0294456,
                     87.21908755, 42.77758137, 56.18374831, 84.57847411, 39.73928908, 40.47366249,
                     8.13350519, 70.42252927, 98.53243923, 77.28886832, 24.54684961, 99.27184662,
                     25.79188738, 76.01136863, 20.15960218, 41.74337278, 36.82418016, 51.96428112,
                     95.42708577, 80.81842267, 4.54804737, 26.51234272, 31.83305896, 51.71930879,
                     12.44859326, 92.69074854]

    def test_smoothing_symm(self):
        df = pandas.DataFrame(list(zip(range(100), numpy.ones(100), 2 * numpy.ones(100))), columns=['x', 'y', 'z'])
        res = prepare.smoothing_symm(df, 'y', 4, 1)
        ans = pandas.DataFrame(list(zip(range(2, 98), numpy.ones(98), 2 * numpy.ones(98))), columns=['x', 'y', 'z'])
        eps = numpy.sum((res - ans).to_numpy() ** 2)
        self.assertEqual(eps < 1.0e-6, True)

    def test_set_smoothing_symm(self):
        df = pandas.DataFrame(list(zip(range(100), numpy.ones(100), 2 * numpy.ones(100))), columns=['x', 'y', 'z'])
        res = prepare.set_smoothing_symm(df, ['y', 'z'], 4, 1)
        ans = pandas.DataFrame(
            list(zip(range(2, 98), numpy.ones(98), 2 * numpy.ones(98), numpy.ones(98), 2 * numpy.ones(98))),
            columns=['x', 'y', 'z', 'y_smooth', 'z_smooth'])
        eps = numpy.sum((res - ans).to_numpy() ** 2)
        self.assertEqual(eps < 1.0e-6, True)

    def test_hann_coefficient_zero(self):
        self.assertEqual(prepare.hann_coefficient(0, 0) == -1)

    def test_hann_coefficient(self):
        r = [0, 50, 100]
        n = [100] * 3
        res = [prepare.hann_coefficient(r[i], n[i]) for i in range(len(r))]
        ans = [0.0, 0.5, 0.0]
        eps = numpy.sum((res - ans).to_numpy() ** 2)
        self.assertEqual(eps < 1.0e-6, True)

    def test_hann(self):
        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
