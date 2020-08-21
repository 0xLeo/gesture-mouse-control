import unittest
import os
import sys
this_script_path = os.path.abspath(__file__)
this_script_folder = os.path.dirname(this_script_path)
sys.path.insert(1, os.path.join(this_script_folder, '..', 'src'))
import numpy as np
from typing import List, Tuple
import cv2
from shapely.geometry import Polygon # Polygon.intersection


class TestHaarCascade(unittest.TestCase):
    def setUp(self):
        self.fpath_casc = os.path.join(this_script_folder, '..', 'cascade', 'hand.xml')
        self.fpath_vid = os.path.join(this_script_folder, 'vid_detection',
                '1_subject_1_room_10_frames.mp4')
        self.fpath_gt = os.path.join(this_script_folder, 'vid_detection',
                '1_subject_1_room_10_frames.csv')
        self._gt_rectangles = None
        self._gt_vid = None
        self._read_ground_truth()

    
    def _read_ground_truth(self):
        """
        # header line 1
        # header line 2
        <video_file_relative_path>
        frame_ind, x0, y0, x1, y1
        """
        if not os.path.isfile(self.fpath_gt):
            print("[DBG] Ground truth %s not found." %
                    os.path.abspath(self.fpath_gt))
            return
        with open(self.fpath_gt) as f:
            gt_lines = f.readlines()
        contains_rectangle = lambda line: '#' not in line and\
            line.count(',') == 4
        contains_fname = lambda line: os.path.isfile(
                os.path.join(this_script_folder, line))
        rect_per_frame = {}
        for line in gt_lines:
            if contains_fname(line):
                self._gt_vid = os.path.join(
                        this_script_folder, line)
            elif contains_rectangle(line):
                vals = [int(n.strip()) for n in line.split(',')]
                ind, x0, y0, x1, y1 = vals[0], vals[1], vals[2], vals[3], vals[4]
                rect_per_frame[ind] = (x0, y0, x1, y1)
        self._gt_rectangles = rect_per_frame
                

    def test_haar_cascade(self, debug = True):
        if not os.path.isfile(self.fpath_casc) or\
        not os.path.isfile(self.fpath_casc) or\
        not os.path.isfile(self.fpath_gt):
            print("[DBG] test_haar_cascade didn't find required files. Skipped.")
            return # there's nothing to test

        hand_detector = cv2.CascadeClassifier(self.fpath_casc)
        cap = cv2.VideoCapture(self.fpath_vid)
        val = True
        fr_ind = 0

        ### test success configs - TODO: move them to an .xml file
        # if IoU > good_IoU, then we have a good detection (good_detections++)
        good_detections, good_IoU = 0, 0.25
        # If good_ious/total_ious < ratio, then the detector is crap and test must fail
        good_IoUs_ratio = 0.2
        IoUs = []
        ### end of config

        while True:
            val, frame = cap.read()
            if not val: # cannot read frame
                break
            try:
                self._gt_rectangles[fr_ind]
            except KeyError:
                continue # no GT for that frame - skip

            else:
                grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                hands = hand_detector.detectMultiScale(grey,
                        scaleFactor = 1.2,
                        minNeighbors = 5,
                        minSize = (64, 64))
                if len(hands) != 0:
                    x0, y0, w, h = hands[0] # pop out the data
                else:
                    x0, y0, w, h = 0, 0, 0, 0

                # find intersection of union (IoU) between GT and measurment
                x1, y1 = x0 + w, y0 + h
                xgt0, ygt0, xgt1, ygt1 = self._gt_rectangles[fr_ind]
                rect = Polygon([(x0,y0), (x0,y1), (x1,y1), (x1,y0)])
                rect_gt = Polygon([(xgt0,ygt0), (xgt0,ygt1), (xgt1,ygt1), (xgt1,ygt0)])
                intersection = rect.intersection(rect_gt).area
                union = rect.area + rect_gt.area - intersection
                IoU = intersection/union
                IoUs.append(IoU)
                if debug:
                    print(x0, y0, x1, y1)
                    print(xgt0, ygt0, xgt1, ygt1)
                    print("Iintersection of union at frame %03d = %.3f" %
                            (fr_ind, IoU))
                    im_lbl = frame.copy()
                    cv2.rectangle(im_lbl, (xgt0, ygt0), (xgt1, ygt1),
                            (255, 60, 0), 3)
                    cv2.rectangle(im_lbl, (x0, y0), (x1, y1),
                            (60, 255, 0), 3)
                    cv2.imshow("GT (blue) vs measurement (green)", im_lbl)
                    cv2.waitKey(3000)
            # update frame counter
            fr_ind += 1
        cv2.destroyAllWindows()
        cap.release()

        if len(IoUs) > 0: # else no data were read so skip
            n_good_IoUs = len([i for i in IoUs if i > good_IoU])
            if debug:
                print("%d good detections over total of %d samples" %
                        (n_good_IoUs, len(IoUs)))
                print("Average IoU (0 < IoU < 1, 1 good) was %.4f" %
                        np.mean(IoUs))
            n_good_IoUs /= len(IoUs)
            self.assertGreater(n_good_IoUs, good_IoUs_ratio)
