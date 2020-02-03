import cv2
import numpy as np

class Heatmap():
    def __init__(self, rows_cols: tuple):
        self.heatmap = np.zeros((rows_cols[0], rows_cols[1]), np.uint8)
        self.max_hits = 4

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
        return int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
