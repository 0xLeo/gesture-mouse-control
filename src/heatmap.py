import cv2
import numpy as np
from collections import deque
import itertools
from typing import Tuple

class Heatmap():
    """Heatmap. A class to accumulate votes given some frames
    and store them on a 2D plane. It finds the peak of that plane.
    """
    def __init__(self, size: Tuple[int, int], max_hits: int = 4, window: int = 7):
        self.heatmap = np.zeros((size[0], size[1]), np.uint8)
        self.max_hits = max_hits
        self.window = window
        # stores the last detected centroids of the hand
        self._last_centroids = deque(maxlen=self.window+1)
        # if the centroid moves less than that, it's considered still
        self._min_moving_displ = 4
        # radii of circles inscribed in detected b.boxes - FILO
        self._rads = [0 for _ in range(self.window)]
        self._rad_mean = None # mean of the above


    def update(self, x0, y0, w, h):
        """update. Update the votes at the heat map (.heatmap).
        Can be used in conjuction with faceCascade.detectMultiScale.
        The latter returns a list of (x, y, w, h) tuple, each one
        can be passed in this method.

        Parameters
        ----------
        x0 :
            top x0 coordinate
        y0 :
            top y0 coordinate
        w :
            width
        h :
            height
        """
        self.heatmap -= 1
        # fix underflow
        self.heatmap[self.heatmap == 255] = 0
        self.heatmap[y0:y0+h, x0:x0+w] += 2
        self.heatmap[self.heatmap > self.max_hits] = self.max_hits
        # update detection radii
        self._rads.pop(0)
        self._rads.append(min(w/2, h/2))


    def find_centroid(self) -> Tuple[int, int]:
        """find_centroid. Find the centroid of the highest values at the map.

        Returns
        -------
        tuple
            The centroid (x, y)
        """
        # find centroid
        thresh_map = np.zeros_like(self.heatmap, np.uint8)
        thresh_map[self.heatmap == self.heatmap.max()] = 255
        M = cv2.moments(thresh_map)
        centr = (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"]))
        self._last_centroids.append(np.array(centr))
        # find mean detectio radius
        self._rad_mean = int(np.mean(self._rads)) 
        return centr, self._rad_mean


    def is_still(self) -> bool:
        """is_still.
        Looks at all the centroids and heuristically determines
        whether the centroid trajectory is stationary.

        Returns
        -------
        bool
            True of stationary, else False
        """
        centrs = self._last_centroids
        if len(centrs) > 1:
            displs = [np.linalg.norm(r1-r2) for r1, r2 in
                    zip(centrs, itertools.islice(centrs, 1, None))]
            return np.median(displs) < self._min_moving_displ
        else:
            return False
