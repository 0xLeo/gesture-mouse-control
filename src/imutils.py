import cv2
import numpy as np


class ImUtils:

    def __init__(self):
        pass
 

    @classmethod
    def qimshow(self, im: np.ndarray, delay: float = 1.0, title = "qimshow", keypress: str = None):
        """qimshow. A wrapper around CV's imshow, waitKey and destroyWindow.
        Shows an image for a certain duration or till a key is pressed

        Parameters
        ----------
        im : np.ndarray
            The image (2D e.g. grey 3D e.g. RGB) to show 
        delay : float
            How long to show the image for. It's ignored if a key is assigned to`keypress`
        title :
            Window title
        keypress : str
            Which key (as a char) to press in order to close the window.
            If None, wait for `delay` seconds.
        """
        if keypress is not None:
            if len(keypress) != 1 or not keypress.isalnum():
                print("Parameter keypress (=%s) must be a single alphanumeric character" %
                        str(keypress))
                return
            keypress = ord(keypress) & 0xff
        key_cur = None
        if keypress is not None:
            while key_cur != keypress:
                cv2.imshow(title, im)
                key_cur = cv2.waitKey(0) & 0xff
        else:
            cv2.imshow(title, im)
            cv2.waitKey(delay=int(delay*1000))
        cv2.destroyWindow(title)
