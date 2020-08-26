import numpy as np
import cv2
from typing import Tuple


class MouseCropper:
    """MouseCropper. Wraps Andrian's code (https://www.pyimagesearch.com/2015/03/09/capturing-mouse-click-events-with-python-and-opencv/) in a class.
<<<<<<< HEAD
    Usage:
    mc = MouseCropper()
    xy0, xy1 = mc.crop(frame)
    mc.close()
=======
>>>>>>> 66c3d0eb5c9163503306e49f8e34aa89f9fcee2e
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
<<<<<<< HEAD
        Wait for second click.
        If both obtained, do nothing.
=======
        Wait for second click. When obtained, exit.
>>>>>>> 66c3d0eb5c9163503306e49f8e34aa89f9fcee2e

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
<<<<<<< HEAD
        if self._first_click and self._second_click:
            return
=======
>>>>>>> 66c3d0eb5c9163503306e49f8e34aa89f9fcee2e
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
<<<<<<< HEAD
            # BUG: if destroyWindow(), then segmentation fault so hide it
            cv2.resizeWindow(self._wname, 2,2)
            cv2.moveWindow(self._wname, 10, 10)


    def _set_window_callback(self, wname: str):
        cv2.namedWindow(wname, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(wname, self._click_callback)


    def crop(self, im: np.ndarray, debug = False) -> Tuple[Tuple[int, int], Tuple[int, int]]:
=======
            cv2.destroyWindow(self._wname)


    def _set_window_callback(self, wname: str):
        cv2.namedWindow(wname)
        cv2.setMouseCallback(wname, self._click_callback)


    def crop(self, im: np.ndarray) -> Tuple[Tuple[int, int], Tuple[int, int]]:
>>>>>>> 66c3d0eb5c9163503306e49f8e34aa89f9fcee2e
        """crop.

        Parameters
        ----------
        im : np.ndarray
            BGR or grey image
<<<<<<< HEAD
        debug :
            Show cropped? (True/False)
=======
>>>>>>> 66c3d0eb5c9163503306e49f8e34aa89f9fcee2e

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
<<<<<<< HEAD
        if debug:
            xy0, xy1 = self._clicks_xy
            cv2.imshow("debug", self._canvas[xy0[1]:xy1[1], xy0[0]: xy1[0]])
            cv2.waitKey(5000)
            cv2.destroyWindow("debug")
        return sorted(self._clicks_xy, key = lambda x: np.linalg.norm(x))


    def close(self):
        """
        Must always be called after crop() method
        """
        cv2.destroyWindow(self._wname)
=======
        return sorted(self._clicks_xy, key = lambda x: np.linalg.norm(x))
>>>>>>> 66c3d0eb5c9163503306e49f8e34aa89f9fcee2e
