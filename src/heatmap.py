import cv2
import numpy as np
from collections import deque
import itertools

class Heatmap():
    def __init__(self, rows_cols: tuple):
        self.heatmap = np.zeros((rows_cols[0], rows_cols[1]), np.uint8)
        self.max_hits = 4
        self._centr_window_len = 5
        # stores the last detected centroids of the hand
        self._last_centroids = deque(maxlen=self._centr_window_len+1)
        # if the centroid moves less than that, it's considered still
        self._min_moving_displ = 5.


    def update(self, x, y, w, h):
        """
        faceCascade.detectMultiScale returns (x, y, w, h)
        Pass (x, y), (x+w, y+h) to this method
        The region inside that rectangle adds +1 to the map

        """
        self.heatmap -= 1
        # fix underflow
        self.heatmap[self.heatmap == 255] = 0
        self.heatmap[y:y+h, x:x+w] += 2
        self.heatmap[self.heatmap > self.max_hits] = self.max_hits


    def find_centroid(self) -> tuple:
        """
        get the centre of the highest area
        """
        thresh_map = np.zeros(self.heatmap.shape, np.uint8)
        thresh_map[self.heatmap == self.heatmap.max()] = 255
        M = cv2.moments(thresh_map)
        centr = (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"]))
        self._last_centroids.append(np.array(centr))
        return centr


    def is_still(self) -> bool:
        """
        Looks at last n centroids and determines if object
        is stationary
        """
        centrs = self._last_centroids
        print(centrs)
        if len(centrs) > 1:
            displs = [np.linalg.norm(r1-r2) for r1,r2 in zip(centrs,\
                    itertools.islice(centrs,1,None))]
            print(displs)
            return np.median(displs) < self._min_moving_displ
        else:
            return False
