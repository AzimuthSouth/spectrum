import unittest
import numpy
from staff import schematisation


class TestSchematisationData(unittest.TestCase):
    def test_merge_trend(self):
        x = numpy.array([[0.0, 0.0],
                         [1.0, 0.5],
                         [2.0, 1.0],
                         [3.0, 1.5],
                         [4.0, 2.0]])
        res = schematisation.merge(x, 1, 1.0)
        ans = numpy.array([[0.0, 0.0],
                           [4.0, 2.0]])
        self.assertEqual(numpy.linalg.norm(res - ans) < 1.0e-7, True)

    def test_merge_local_extreme(self):
        x = numpy.array([[0.0, 0.0],
                         [1.0, 3.0],
                         [2.0, 2.0],
                         [3.0, 3.0],
                         [4.0, 0.0]])
        res = schematisation.merge(x, 1, 1.5)
        # print(res)
        ans = numpy.array([[0.0, 0.0],
                           [1.0, 3.0],
                           [3.0, 3.0],
                           [4.0, 0.0]])
        #self.assertEqual(numpy.linalg.norm(res - ans) < 1.0e-7, True)

