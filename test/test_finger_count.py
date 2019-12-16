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
        # create auxiliary members
        test_path = os.path.join('..', 'test_images', 'count_fingers')
        files_jpg = os.path.join(test_path, '*.jpg')
        files_png = os.path.join(test_path, '*.png')
        file_gt = os.path.join(test_path, 'ground_truth.txt')
        self.files_to_read = glob.glob(files_jpg) + glob.glob(files_png)
        assert(len(self.files_to_read) and os.path.isfile(file_gt)),\
            'Directory %s must contain image1, image2,... and'\
            'a file ground_truth.txt -> <image_file> <no_fingers>'\
            % test_path 
        self.ground_truth = {}
        with open(file_gt) as f:
            for l in f.readlines():
                self.ground_truth[l.strip().split(' ')[0]] =\
                        int(l.strip().split(' ')[1])


    def tearDown(self):
        # delete auxiliary members
        pass


    def read_image(self, fpath):
        '''
        Auxiliary method
        '''
        im = cv2.imread(fpath)
        im = cv2.GaussianBlur(im, (11, 11), 1.2, 1.2) 
        hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        return hsv


    def test_finger_counter(self):
        '''
        Counts number of fingers and compares against GT
        '''
        # TODO: count fingers, if good accuracy pass tests
        results = {}
        for f in self.files_to_read:
            hsv = self.read_image(f)
            km = KMeans(hsv, k = 3)
            km.do_kmeans()
            km.find_most_frequent()
            # key of self.ground_truth
            fname = f.split(os.sep)[-1]
            n = km.count_fingers()
            results[f] = (n, self.ground_truth[fname] == n)
        total_correct = len([r[1] for r in results.values() if r[1]])
        print("%s: %d/%d correct results" %\
            (self.test_finger_counter.__name__,total_correct, len(results)))
        # 3 possible classfications.
        # Perform better than random classifier? Then pass.
        self.assertGreater(total_correct, len(results)/3)


if __name__ == '__main__':
    unittest.main()
