import numpy as np
import cv2
from collections import deque
from functools import reduce


class DenseOptFlow:
    """DenseOptFlow."""
    def __init__(self, history = 3):
        """__init__.

        Parameters
        ----------
        history :
            How many magnitudes to store when combuting the combined flow
        """
        # FILO that stores the magnitude matrices outputted by opt flow
        self._flow_ring_buffer = deque()
        self._history = history
        self._max = None
        self._max_mag = None


    def find_flow(self, im1: np.ndarray, im2: np.ndarray, blur = True, debug = False):
        """find_flow. Finds optical flow between previous and current frame

        Parameters
        ----------
        im1 : np.ndarray
            previous frame as BGR or grey image
        im2 : np.ndarray
            current frame as BGR or grey image
        blur :
            Apply Gaussian blurring to both frames?
        debug :
            Show debug output (optical flow)?
        """
        if len(im1.shape) == 3:
            im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
        if len(im2.shape) == 3:
            im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
        if blur:
            im1 = cv2.GaussianBlur(im1, (13, 13), sigmaX = 2, sigmaY = 2)
            im2 = cv2.GaussianBlur(im2, (13, 13), sigmaX = 2, sigmaY = 2)
        flow = cv2.calcOpticalFlowFarneback(im1, im2,
                None, 0.5, 3, 15, 3, 5, 1.2, 0)
        # angle starts from 0 and ends at 2*pi
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        self._max_mag = np.max(mag)
        if debug:
            # create an HSV image to visualise flow by mapping
            # angle matrix values from 0 to 180, and mag from 0 to 255
            hsv_mask = np.zeros((im1.shape[0], im1.shape[1], 3), np.uint8)
            hsv_mask[..., 0] = ang * 180 / np.pi / 2 # hue (H)
            hsv_mask[..., 1] = 255 # saturation (S)
            hsv_mask[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            bgr_mask = cv2.cvtColor(hsv_mask, cv2.COLOR_HSV2BGR)
            cv2.imshow('optical flow', bgr_mask)
            cv2.waitKey(33)
        self._mag = mag
        return mag, ang


    def get_combined_flow(self) -> np.ndarray:
        """ Looks into the ring buffer elements (magnitude matrices) and
        returns a matrix that represents their combination
        """
        # convert it to 8-bit image
        self._mag = cv2.normalize(self._mag, None, 0, 255, cv2.NORM_MINMAX)
        self._mag = np.array(self._mag, np.uint8)
        _, self._mag = cv2.threshold(self._mag, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if len(self._flow_ring_buffer) == 0:
            # initialise the ring buffer
            for _ in range(self._history):
                self._flow_ring_buffer.append(self._mag)
        else:
            # update ring buffer
            self._flow_ring_buffer.popleft()
            self._flow_ring_buffer.append(self._mag)
        # OR all elements together
        return reduce(lambda x, y: np.bitwise_or(x, y), self._flow_ring_buffer)
