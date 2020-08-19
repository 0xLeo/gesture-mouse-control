import cv2
import numpy as np
from typing import Union, List, Tuple


def qimshow(im: np.ndarray, delay: float = 1.0, title = "qimshow", keypress: str = None):
    """qimshow. Quick imshow. A wrapper around CV's imshow, waitKey and destroyWindow.
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
            print("Parameter keypress (=%s) must be a single"
                    "alphanumeric character" % str(keypress))
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



def text_on_im(im: np.ndarray,
        text: str,
        pos: Union[str, Tuple[int, int]] = 'tr',
        font: str = cv2.FONT_HERSHEY_SIMPLEX,
        col: Tuple[int, int, int] = (70, 255, 50),
        thickness = 1):
    """text_on_im.

    Parameters
    ----------
    im : np.ndarray
        greyscale or BGR image
    text : str
        text to write on image
    pos : Union[str, Tuple[int, int]]
        position where text shall start
        'tr' for top right, 'tl' for top left, else (x, y) num. tuple
    font : str
        do [f for f in dir(cv2) if 'FONT' in f]  to view options
    col : Tuple[int, int, int]
        (B, G, R) letter colour
    thickness :
        letter thickness

    Returns
    -------
    np.ndarray
        A BGR image with coloured text on it
    """
    size = 0.7 # fixed size to break lines properly
    im_cpy = im.copy()
    if len(im_cpy.shape) == 2: # to BGR for coloured text
        im_cpy = cv2.cvtColor(im_cpy, cv2.COLOR_GRAY2BGR)
    height, width, _ = im_cpy.shape
    px_padding = 30

    # make sure position is in coordinates
    if pos == 'tr': # top right
        pos = (int(width/2), px_padding)
    elif pos == 'tl': # top left
        pos = (px_padding, px_padding)
   
    l_height = 25
    # CV doesn't support newline in putText so do it manually
    for i, text_line in enumerate(text.split('\n')):
        cv2.putText(im_cpy, text=text_line, org=(pos[0], pos[1]+i*l_height),
            fontFace=font, fontScale=size, color=col, thickness=thickness)
    return im_cpy



def draw_point(im: np.ndarray, pt: Tuple[int, int],
        col: Tuple[int, int, int] = (50, 255, 40)) -> np.ndarray:
    """draw_point. A wrapper around CV's `circle`.

    Parameters
    ----------
    im : np.ndarray
        grey or BGR image
    pt : Tuple[int, int]
        where to draw the dot in (x, y) coordinates
    col : Tuple[int, int, int]
        8-bit (B, G, R) colour triplet

    Returns
    -------
    np.ndarray
        a BGR image width a dot drawn on it
    """
    im_cpy = im.copy()
    if len(im_cpy.shape) == 2: # to BGR for coloured circle
        im_cpy = cv2.cvtColor(im_cpy, cv2.COLOR_GRAY2BGR)
    rad = 3
    thickness = 2
    return cv2.circle(im_cpy, pt, rad, col, thickness)


def imscale(im: np.ndarray, scale: float) -> np.ndarray:
    """imscale. Scales an image down/up by a factor.

    Parameters
    ----------
    im : np.ndarray
        BGR or grey image
    scale : float
        factor to scale by

    Returns
    -------
    np.ndarray
        Up/downscaled image
    """
    dims = im.shape
    h, w = dims[0], dims[1]
    h, w = int(w * scale), int(h * scale)
    im = cv2.resize(im, (h, w))
    return im
