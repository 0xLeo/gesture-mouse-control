import unittest
import os
import sys
this_script_path = os.path.abspath(__file__)
this_script_folder = os.path.dirname(this_script_path)
sys.path.insert(1, os.path.join(this_script_folder, '..', 'src'))
from imutils import ImUtils
import numpy as np
import time


class TestImUtilsNoKeypress(unittest.TestCase):
    def setUp(self):
        self.im_to_show = np.array(np.eye(200)*255, np.uint8) # diag line


    def test_class(self):
        ImUtils.qimshow(self.im_to_show)


    def test_object(self):
        iu = ImUtils()
        iu.qimshow(self.im_to_show)


    def test_duration(self):
        duration = 4 # in sec
        time_st = time.time()
        ImUtils.qimshow(self.im_to_show, delay = duration) 
        self.assertEqual( 3.5 < time.time() - time_st < 4.5, True) # about 4 sec
