#!/usr/bin/env python3
import sys
import os
import cv2
import numpy as np
this_script_path = os.path.abspath(__file__)
this_script_folder = os.path.dirname(this_script_path)
sys.path.insert(1, os.path.join(this_script_folder, '..', 'src'))
from heatmap import Heatmap
from myimutils import qimshow, text_on_im, draw_point, imscale
from dense_opt_flow import DenseOptFlow


fpath_casc = '../cascade/hand.xml'
fpath_vid = '../tests/vid_detection/sample_01.mp4'

hand_detector = cv2.CascadeClassifier(fpath_casc)
cap = cv2.VideoCapture(fpath_vid)
_, frame1 = cap.read()
first = True
x0, y0, w, h = 0, 0, 0, 0
dof = DenseOptFlow()

while True:
    val, frame = cap.read()
    #frame  = imscale(frame, 0.5)
    if not val:
        pass
        #break
    if first:
        r, c, _ = frame.shape
        hm = Heatmap((r, c))
        first = False
        prvs, curr = frame.copy(), frame.copy()
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    curr = grey
    flow_mag, flow_ang = dof.find_flow(prvs, curr)
    flow_mag = dof.get_combined_flow()
    hands = hand_detector.detectMultiScale(grey,
            scaleFactor = 1.15,
            minNeighbors = 5,
            minSize = (64, 64))
    print(hands)
    for hand in hands:
        try:
            x0, y0, w, h = hand
            hm.update(x0, y0, w, h)
        except:
            pass
    x1, y1 = x0 + w, y0 + h
    r1 = int(w/2*1.1)
    r2 = int(2.1*r)
    r = r1
    centroid = hm.find_centroid()

    #fr_show = cv2.rectangle(frame.copy(), (x0, y0), (x1, y1), color = (60,255,0), thickness = 3)
    fr_show = draw_point(flow_mag.copy(), centroid)
    fr_show = cv2.circle(fr_show, centroid, r, (60,255,0), 3)
    fr_show = cv2.circle(fr_show, centroid, int(2.1*r), (60,255,0), 3)
    if hm.is_still():
        fr_show = text_on_im(fr_show, "STILL", thickness = 2, col = (70,255,0))
    cv2.imshow("", fr_show)
    cv2.waitKey(33)
    print(x0, y0, x1, y1)
    prvs = curr 

cv2.destroyAllWindows()
