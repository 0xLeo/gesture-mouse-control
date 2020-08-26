import numpy as np
import cv2
from typing import Tuple


class MouseCropper:
    """MouseCropper. Wraps Andrian's code (https://www.pyimagesearch.com/2015/03/09/capturing-mouse-click-events-with-python-and-opencv/) in a class.
    """

    def __init__(self):
        self._clicks_xy = []
        self._wname = "to crop"
        self._first_click = False
        self._second_click = False
        self._canvas = None


    def _click_callback(self, event, x: int, y: int, flags, param):
        """_click_callback. Wait for first click and store it.
        Get mouse position and draw rectangle from first click to position.
        Wait for second click. When obtained, exit.

        Parameters
        ----------
        event :
            event
        x :
            x coordinate
        y :
            y coordinate
        flags :
            callback unused parameter
        param :
            callback unused parameter
        """
        if event == cv2.EVENT_MOUSEMOVE:
            if not self._first_click:
                return
            else:
                cv2.imshow(self._wname, cv2.rectangle(self._canvas.copy(), (self._clicks_xy[0]), (x,y), (60, 255,0), 3))
        elif event == cv2.EVENT_LBUTTONDOWN:
            if not self._first_click:
                self._clicks_xy.append((x,y))
                self._first_click = True
            elif not self._second_click:
                self._clicks_xy.append((x,y))
                self._second_click = True
        if self._first_click and self._second_click:
            cv2.destroyWindow(self._wname)


    def _set_window_callback(self, wname: str):
        cv2.namedWindow(wname)
        cv2.setMouseCallback(wname, self._click_callback)


    def crop(self, im: np.ndarray) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """crop.

        Parameters
        ----------
        im : np.ndarray
            BGR or grey image

        Returns
        -------
        Tuple[Tuple[int, int], Tuple[int, int]]
            The rectangle top left and bottom right vertices as (x0,y0), (x1,y1)
            sorted w.r.t their distance from origin (top left).
        """
        if len(im.shape) == 2:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        self._canvas = im.copy()
        self._set_window_callback(self._wname)
        cv2.imshow(self._wname, im)
        cv2.waitKey(0)
        return sorted(self._clicks_xy, key = lambda x: np.linalg.norm(x))
