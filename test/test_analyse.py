import unittest
import numpy
import scipy

from staff import analyse
from staff import prepare


class TestAnalyse(unittest.TestCase):
    def test_spectrum_density_harmonic(self):
        t = numpy.arange(0, 10, 0.01)
        f0 = 2.0
        a = 3.0
        x = a * numpy.cos(2 * numpy.pi * f0 * t)
        f, g = analyse.spectrum_density(x, len(x), len(x))
        fm = f[numpy.where(g == max(g))][0]
        self.assertEqual((max(g) - a ** 2 / 2) < 1.0e-7 and (f0 - fm < 1.0e-7), True,
                         "spectrum density failed")

    def test_n_coefficient(self):
        z = [1.0 + 1.0j, 1.0, -1.0, -1.0 + 1.0j, -1.0 - 1.0j, 1.0j, -1.0j]
        res = [analyse.n_coefficient(k) for k in z]
        ans = [0.0, 0.0, 1.0, 1.0, -1.0, 0.0, 0.0]
        eps = 0
        for i in range(len(z)):
            eps += (res[i] - ans[i]) ** 2
        self.assertEqual(eps ** 0.5 < 1.0e-7, True, "n_coefficient test failed")

    def test_mod_phase(self):
        z = [2.0 ** 0.5, 1.0 + 1.0j, 2 ** 0.5 * 1.0j, -1.0 + 1.0j, -2.0 ** 0.5, -1.0 - 1.0j, -2 ** 0.5 * 1.0j,
             1.0 - 1.0j]
        [m, f] = analyse.cross_spectrum_mod_fas(z)
        mod = numpy.full(len(z), 2 ** 0.5)
        fas = numpy.asarray([0.0, numpy.pi / 4, numpy.pi / 2, 3 * numpy.pi / 4, numpy.pi,
                             -3 * numpy.pi / 4, -numpy.pi / 2, -numpy.pi / 4])
        eps1 = numpy.linalg.norm(mod - m)
        eps2 = numpy.linalg.norm(fas - f)
        self.assertEqual(eps1 < 1.0e-7 and eps2 < 1.0e-7, True, "module or phase calculation failed")

    def test_frequencies(self):
        res = analyse.frequencies(100, 0.01)
        ans = scipy.fft.rfftfreq(100, 0.01)
        eps = numpy.sum((res - ans) ** 2)
        self.assertEqual(eps ** 0.5 < 1.0e-7, True, "frequencies test failed")


if __name__ == '__main__':
    unittest.main()
