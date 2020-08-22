import unittest
import os
import sys
this_script_path = os.path.abspath(__file__)
this_script_folder = os.path.dirname(this_script_path)
sys.path.insert(1, os.path.join(this_script_folder, '..', 'src'))
from dense_opt_flow import DenseOptFlow
import numpy as np


class TestDenseOptFlow(unittest.TestCase):
   
    def setUp(self):
        size = (240, 320)
        self.im_prev = np.random.randint(0, 255, size = size, dtype = np.uint8)
        self.im_curr = np.random.randint(0, 255, size = size, dtype = np.uint8)


    def test_dense_opt_flow_outputs(self):
        """test return sizes and values"""
        dof = DenseOptFlow()
        mag, ang = dof.find_flow(self.im_prev, self.im_curr, blur=True)

        cond1 = np.all(0 <= ang) and np.all (ang <= 2*np.pi)
        cond2 = mag.shape == self.im_prev.shape
        cond3 = ang.shape == self.im_prev.shape
        self.assertEqual(cond1, True)
        self.assertEqual(cond2, True)
        self.assertEqual(cond3, True)
