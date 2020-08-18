import unittest
import os
import sys
this_script_path = os.path.abspath(__file__)
this_script_folder = os.path.dirname(this_script_path)
sys.path.insert(1, os.path.join(this_script_folder, '..', 'src'))
from imutils import ImUtils
import numpy as np
import time
import cv2


class TestImUtilsNoKeypress(unittest.TestCase):
    def setUp(self):
        self.im_to_show = np.array(np.eye(200)*255, np.uint8) # diag line


    def test_class(self):
        ImUtils.qimshow(self.im_to_show)


    def test_object(self):
        iu = ImUtils()
        iu.qimshow(self.im_to_show)


    def test_duration(self):
        duration = 3 # in sec
        time_st = time.time()
        ImUtils.qimshow(self.im_to_show, delay = duration)
        # should last about `duration` seconds
        self.assertEqual( duration - .5 < time.time() - time_st < duration + .5, True)


class TestImUtilsText(unittest.TestCase):
    def setUp(self):
        self.im_grey = np.zeros((300, 500), np.uint8)
        self.im_bgr = np.zeros((300, 500, 3), np.uint8)
        # choose something short to test the location etc.
        self.text = 'test'


    def test_text(self):
        """
        Write some coloured text in the first quarterion of the image.
        Check whether its centre is there and whether it's coloured.
        """
        im = self.im_grey
        h, w = im.shape
        im_text = ImUtils.text_on_im(im, self.text, pos=(int(w/2), 30))
        im_text_grey = cv2.cvtColor(im_text, cv2.COLOR_BGR2GRAY)

        # black image with (B, G, R) letters -> [0, B, G, R] list
        colour_vals = np.unique(im_text)
        yx_nz = np.where(im_text_grey != 0)
        x_nz, y_nz = yx_nz[1], yx_nz[0]
        x_centre, y_centre = np.mean(x_nz), np.mean(y_nz)
        #ImUtils.qimshow(im_text)

        self.assertEqual(x_centre > w/2 and y_centre < h/2, True)
        # 0 + the letter colour triplet
        self.assertEqual(len(colour_vals), 4)
