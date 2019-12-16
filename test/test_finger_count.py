#!/usr/bin/env python

import unittest
import sys
sys.path.insert(1, '../src')
from kmeans import KMeans 
import cv2
import os
import glob


class TestKMeans(unittest.TestCase):
    def setUp(self):
        # create auxilary members
        self.files_to_read = glob.glob('../test/count_fingers/*.jpg') +\
            glob.glob('../test/count_fingers/*.png')

    def tearDown(self):
        # delete auxilary members
        pass

    def read_image(self, fpath):
        '''
        Auxiliary method
        '''
        im = cv2.imread(fpath)
        im = cv2.GaussianBlur(im, (11, 11), 1.2, 1.2) 
        hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        return hsv

    def test_create_skin_detector(self):
        '''
        Require skin detector to have some hsv_low and hsv_high
        limits, ideally set to either None or some default value.
        '''
        # TODO: count fingers, if good accuracy pass tests
        for f in self.files_to_read:
            hsv = self.read_image(f)
        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
