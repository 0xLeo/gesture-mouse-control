import unittest
import os
import sys
this_script_path = os.path.abspath(__file__)
this_script_folder = os.path.dirname(this_script_path)
sys.path.insert(1, os.path.join(this_script_folder, '..', 'src'))
import heatmap 
import numpy as np
import cv2


class TestHaarCascade(unittest.TestCase):
    def setUp(self):
        self.fpath_casc = os.path.join(this_script_folder, '..', 'cascade', 'hand.xml')
        self.fpath_vid = os.path.join(this_script_folder, 'vid_detection',
                '1_subject_1_room_10_frames.mp4')
        self.fpath_gt = os.path.join(this_script_folder, 'vid_detection',
                '1_subject_1_room_10_frames.csv')

    
    def test_haar_cascade(self, debug = False):
        if not os.path.isfile(self.fpath_casc) or\
        not os.path.isfile(self.fpath_casc) or\
        not os.path.isfile(self.fpath_gt):
            print("[DBG] test_haar_cascade didn't find required files. Skipped.")
            return # there's nothing to test
        hand_detector = cv2.CascadeClassifier(self.fpath_casc)
        cap = cv2.VideoCapture(self.fpath_vid)
        val, first = True, True
        while True:
            val, frame = cap.read()
            if not val:
                break
            if first:
                r, c, _ = frame.shape
                hm = heatmap.Heatmap((r, c))
                first = False
            else:
                grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                hands = hand_detector.detectMultiScale(grey,
                        scaleFactor = 1.2,
                        minNeighbors = 5,
                        minSize = (64, 64))
                x, y, w, h = hands[0][0], hands[0][1], hands[0][2], hands[0][3]
                if debug:
                    cv2.imshow("", cv2.rectangle(frame, (x,y), (x+w, y+h),
                        (60, 255, 0), 3))
                    cv2.waitKey(33)
        cap.release()
        cv2.destroyAllWindows()


