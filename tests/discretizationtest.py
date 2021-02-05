import unittest
from unittest import TestCase
import pandas as pd
import numpy as np
import sys
sys.path.append('../')
import fairsd.discretization as disc

class TestMDLP(TestCase):
    def test_findCutPoints(self):
        a=pd.read_csv("age.csv")
        x=a["age"]
        y=a["y_true"]
        discretizer = disc.MDLP()
        correct_result=[21.0, 23.0, 24.0, 27.0, 30.0, 35.0, 41.0, 54.0, 61.0, 67.0]
        self.assertEqual(discretizer.findCutPoints(x, y), correct_result)

class TestEqualFrequency(TestCase):
    def test_findCutPoints(self):

        discretizer = disc.EqualFrequency(2, 0)

        x = np.ones(49)
        x = np.concatenate((x, [2]))  # no cut points expected because of the min_bin_size
        self.assertEqual(discretizer.findCutPoints(x), [])
        x = np.concatenate((x, [2]))  # expected one cut point
        self.assertEqual(discretizer.findCutPoints(x), [1])
        x = np.concatenate((x, [2]))  # expected one cut point
        self.assertEqual(discretizer.findCutPoints(x), [1])

        x = np.ones(5)
        correct_res = []
        self.assertEqual(discretizer.findCutPoints(x), correct_res) #no cut points expected

        x = np.concatenate((x, np.zeros(5)))
        correct_res.append(0)
        self.assertEqual(discretizer.findCutPoints(x), correct_res)

        x=np.array([1,2,3,4,5,6,7,8,9,10])
        self.assertEqual(discretizer.findCutPoints(x), [3,5,8])
        discretizer = disc.EqualFrequency(1, 4)
        self.assertEqual(discretizer.findCutPoints(x), [3,5,8])
        discretizer = disc.EqualFrequency(1, 0)
        self.assertEqual(discretizer.findCutPoints(x), [2, 3, 4, 5, 7, 8, 9])
        x = np.array([1, 2, 2, 2, 2, 6, 7, 8, 9, 10])
        self.assertEqual(discretizer.findCutPoints(x), [2, 7, 8, 9])
        
        discretizer = disc.EqualFrequency(2, 0)
        x = np.array([1,7,7,7,7,7,7,8,7,7])
        self.assertEqual(discretizer.findCutPoints(x), [])


class TestEqualWidth(TestCase):
    def test_findCutPoints(self):
        discretizer = disc.EqualWidth(2,20)
        x=np.zeros(9)
        x=np.concatenate((x, [10]))
        self.assertEqual(discretizer.findCutPoints(x), [2.5,5,7.5])
if __name__ == '__main__':
    unittest.main()