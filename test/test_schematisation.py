import unittest
import numpy
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
        res = schematisation.merge(self.x1, 1, 1.0)
        ans = numpy.array([[0.0, 0.0], [4.0, 2.0]])
        self.assertEqual(numpy.linalg.norm(res - ans) < 1.0e-7, True)

    def test_merge_local_extreme(self):
        self.setdata()
        res = schematisation.merge(self.x2, 1, 1.5)
        # print(res)
        ans = numpy.array([[0.0, 0.0], [1.0, 3.0], [3.0, 3.0], [4.0, 0.0]])
        self.assertEqual(numpy.linalg.norm(res - ans) < 1.0e-7, True)

    def test_merge_25(self):
        self.setdata()
        res = schematisation.merge(self.x3, 1, 0.9)
        ans = numpy.array([[0.0, 6.0],
                           [1.0, 12.0],
                           [2.0, 10.0],
                           [3.0, 11.0],
                           [4.0, 8.0],
                           [5.0, 6.0],
                           [11.0, 4.0],
                           [12.0, 7.0],
                           [13.0, 10.0],
                           [14.0, 11.0],
                           [16.0, 11.0]])
        self.assertEqual(numpy.linalg.norm(res - ans) < 1.0e-7, True)

    def test_extreme_count(self):
        self.setdata()
        res = numpy.array([schematisation.extreme_count(self.x1[:, 1]), schematisation.extreme_count(self.x2[:, 1]),
                           schematisation.extreme_count(self.x3[:, 1])])
        ans = numpy.array([0, 3, 7])
        self.assertEqual(numpy.linalg.norm(res - ans) < 1.0e-7, True)
