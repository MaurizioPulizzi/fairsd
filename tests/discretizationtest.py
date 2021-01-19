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

if __name__ == '__main__':
    unittest.main()