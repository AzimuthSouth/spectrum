import unittest
import numpy

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

    def test_smoothing(self):
        self.setdata()
        res = prepare.smoothing(self.data, 4)
        ans = [-1.36099342e+01, 2.04275193e+01, 3.13955208e+01, 2.25645194e+01,
               - 1.99317865e+01, 2.66722227e-01, - 3.49724515e+01, 7.51742529e+00,
               3.10875921e+01, - 9.49368396e+00, - 1.53526340e+01, - 3.64885727e+01,
               8.64920990e+00, 1.42248898e+01, - 1.00990230e+01, 4.35615445e+01,
               9.31397729e+00, - 3.93683272e+01, - 3.74110832e+01, 1.23380805e+01,
               1.38037966e+01, - 2.14006664e+00, 5.31705708e-01, 5.97697990e+00,
               1.95293647e+01, - 1.30421919e+01, 9.39954815e-01, 4.13472414e+01,
               4.70425735e-02, - 1.39168716e+01, - 5.54608303e+01, 2.72485766e+00,
               2.36224383e+01, 2.05640053e+01, - 3.18586384e+01, 4.39631704e+01,
               - 1.51346704e+01, 3.23267377e+01, - 1.75132569e+01, - 1.47463572e+01,
               - 2.94343123e+01, - 6.22517812e+00, 4.36006111e+01, 4.48904547e+01,
               - 2.41051421e+01, - 4.11598321e+00, - 1.53398684e+01, 5.17193088e+01,
               1.24485933e+01, 9.26907485e+01]

        self.assertEqual(numpy.linalg.norm(res - ans) < 1.0e-6, True)

    def test_hann(self):
        self.setdata()
        res = prepare.correction_hann(prepare.smoothing(self.data, 4))
        ans = [-0.0, 0.08385482543440552, 0.5133974031852788, 0.8245441227458384,
               -1.2824240576553159, 0.026483700484517467, -4.925006852239974, 1.4151936420349525,
               7.4860076895920615, -2.825399412791489, -5.492193061381769, -15.332500154851896,
               4.185994537446609, 7.795403385822147, -6.173133494066359, 29.30308984316876,
               6.81102421335977, -30.94580151445351, -31.281293869653997, 10.866431006010258,
               12.686297155559567, -2.034100024415491, 0.5181620325824342, 5.921869622829765,
               19.509302124589883, -13.02879356035574, 0.9312880353933057, 40.29403923139346,
               0.04471323385800682, -12.790218032186699, -48.8456277021665, 2.2783909479332443,
               18.568614391722967, 15.037822602410554, -21.43074942577372, 26.872948077778602,
               -8.293973602444284, 15.645307375239335, -7.359071458449661, -5.275305878886732,
               -8.759896472712153, -1.499046021895888, 8.20803736865746, 6.3217128759650585,
               -2.3934764308597005, -0.2648250268085467, -0.5605436633797664, 0.8457435388225538,
               0.0511013892784202, 0.0]
        eps = []
        for i in range(len(res)):
            eps.append(res[i] - ans[i])
        self.assertEqual(numpy.linalg.norm(eps) < 1.0e-6, True)

    def draw_weight(self):
        x = numpy.arange(100)
        y = numpy.ndarray((4, len(x)))
        for i in range(len(x)):
            y[0][i] = prepare.hann_coefficient(i, 100)
            y[1][i] = prepare.triangle_coefficient(i, 100)
            y[2][i] = prepare.hemming_coefficient(i, 100)
            y[3][i] = prepare.natoll_coefficient(i, 100, 0.8, 0.1, 0.1)
        p = draw.Painter()
        p.draw(x, y, ["hann_koef", "treug_coef", "hemming", "natoll"], ["time", "data"])


if __name__ == '__main__':
    unittest.main()