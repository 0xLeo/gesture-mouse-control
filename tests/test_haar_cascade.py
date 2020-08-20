import unittest
import os
import sys
this_script_path = os.path.abspath(__file__)
this_script_folder = os.path.dirname(this_script_path)
sys.path.insert(1, os.path.join(this_script_folder, '..', 'src'))
import myimutils
import numpy as np
import cv2


class TestHaarCascade(unittest.TestCase):
    def setUp(self):
        self.fpath_casc = os.path.join('..', 'cascade', 'hand.xml')
        self.fpath_vid = os.path.join('vid_detection',
                '1_subject_1_room_10_frames.mp4')
        if not os.path.isfile(self.fpath_casc) or not os.path.isfile(self.fpath_casc):
            return # there's nothing to test

    
    def test_haar_cascade(self, debug = False):
        hand_detector = cv2.CascadeClassifier(self.fpath_casc)
        cap = cv2.VideoCapture(self.fpath_vid)
        val, first = True, True
        while val:
            val, frame = cap.read()
            if first:
                r, c, _ = frame.shape
                hm = Heatmap((r, c))
                first = False
            else:
                grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                hands = hand_detector.detectMultiScale(grey,
                        scaleFactor = 1.2,
                        minNeighbors = 5,
                        min_size = (64, 64))
                if debug:
                    cv2.imshow("", grey)
                    cv2.waitKey(33)
        cap.release()
        cv2.destroyAllWindows()


