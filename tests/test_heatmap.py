import unittest
import os
import sys
this_script_path = os.path.abspath(__file__)
this_script_folder = os.path.dirname(this_script_path)
sys.path.insert(1, os.path.join(this_script_folder, '..', 'src'))
import heatmap
import numpy as np
import cv2


class TestHeatmap(unittest.TestCase):
    def setUp(self):
        pass


    def test_find_centroid(self):
        """              
        Find heatmap centroid for the simplest case of 2 overlapping rectangles
        of votes.

        +----------------+
        |1111111111111111|
        |1111111111111111|
        |111111+---------+-----------+
        |111111|222222222|11111111111|
        |111111|2222o2222|11111111111|
        |111111|2222|2222|11111111111|
        +------+----|----+11111111111|
               |1111|1111111111111111|
               |1111|1111111111111111|
               +----|----------------+
                    |
                    +-------> (x, y)
        """
        rows, cols = 240, 320
        # x0, y0, w, h
        rects = [(260, 100, 20, 30), (270, 110, 40, 50)]
        ### find centroid using the class
        hm = heatmap.Heatmap((rows, cols))
        for rect in rects:
            hm.update(*rect)
        centroid = hm.find_centroid()
        ### find centroid manually
        canvas = np.zeros((rows, cols), np.uint8)
        for rect in rects:
            x, y, w, h = rect
            canvas[y:y+h, x:x+w] += 1 # np index: rows, cols
        canvas_max = np.zeros_like(canvas)
        canvas_max[canvas == np.max(canvas)] = 1
        M = cv2.moments(canvas_max)
        centroid_correct = (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"]))
        ### validate
        self.assertEqual(all([p == q for p, q in zip(centroid, centroid_correct)]),
                True)
