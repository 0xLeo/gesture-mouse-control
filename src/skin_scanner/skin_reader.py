#!/usr/bin/python3
import cv2
import time
import numpy as np
import pickle

COLOR_WHITE = (255, 255, 255)
COLOR_GREEN = (0, 255, 0)

def qimshow(im, delay = 5, wname = 'display',  close = True):
    cv2.imshow(wname, im)
    if close:
        cv2.waitKey(int(delay * 1000))
        cv2.destroyAllWindows()

def text_on_image(text, im):
    y0,  dy = 30,  30
    for i, line in enumerate(text.split('\n')):
        y = y0 + i*dy
        cv2.putText(im, line, (50, y ), cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLOR_WHITE, 2)
    return im 


def accum_hs_hist(cam,  bgr, top, bottom, mirror = True, timeout = 10):
    """
    This function reads from the camera descriptor (1st argument), 
    and crops a certain area of each frame. It extracts a
    representative skin tone for each human from this area.

    @param cam: camera descriptor as returned from cv2.VideoCapture
    @param bgr: the captured frame in BGR
    @param top: a tuple of (row, col) screen coordinates  
    @param bottom: another (row, col) tuple, together with top they  
        define the bounding box whose interior extract the skin tone
    @param mirror: horizontally flip the frame? 
    @param timeout: how long to scran the skin tone for (seconds) 
    @pickle histogram_sum: the HS histogram of the cropped frame.
        It defines the range of the detected skin tone. Exported
        as a pickle object in file histogram_sum. Contains a 180x
        255 float array.
    @return: hs_hist_sum the HS histogram sum as a variable 
    """
    cv2.destroyAllWindows()
    # sample generic unrefined skin tone range from pyimagesearch.com
    lower = np.array([0, 48, 80], dtype = "uint8")
    upper = np.array([20, 255, 255], dtype = "uint8")
    hs_hist_sum = np.zeros((180, 256), np.float32)
    
    print()
    print('Scanning begins... press s if you want to exit...')
    time.sleep(2)
    valid = True
    time_start = time.time()
    while valid and time.time() - time_start < timeout:
        valid, im = cam.read()
        if mirror: 
            im = cv2.flip(im, 1)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        skinMask = cv2.inRange(hsv, lower, upper)
        # get skin only
        hsv = cv2.bitwise_and(hsv, hsv, mask = skinMask)
        im_skin = cv2.bitwise_and(im, im, mask=skinMask) 
        hsv_cropped = hsv[top[0]:bottom[0],  top[1]:bottom[1]]
        hs_hist = cv2.calcHist([hsv_cropped], [0, 1], None, [180, 256], [0, 180, 0, 256])
        hs_hist = hs_hist / np.sum(hs_hist)
        hs_hist_sum += hs_hist
        cv2.rectangle(im,  top , bottom,  COLOR_GREEN, 4)
        cv2.imshow('skin scanner', im)
        if cv2.waitKey(10) == ord('s'):
            break
    hs_hist_sum = np.uint64(hs_hist_sum)
    with open('histogram_sum',  'wb') as f:
        pickle.dump(hs_hist_sum, f)
    return hs_hist_sum

def extract_skin_range(pickle_file = 'histogram_sum', strict = 5):
    """
    This function reads the pickle output of the histogram extracter
    (accum_hs_hist) and decodes the HS range of the skin range.
    @param pickle_file: The pickle file you want to extract the skin
        range from. Should be a 180x255 float array.
    @param script: Takes values from 1, 2, ..., 5. The higher the
        strictness, the closer the skin tone to the refined one.
        The lower, the closer to the generic limits.
    @pickle hsv_refined_lower: Reads input pickle and extracts the
        minimum limit of the HSV skin values as [h_min, s_min, 80]
    @pickle hsv_refined_upper: Similar, as [h_max, s_max, 255]
    """
    lower = np.array([0, 48, 80], dtype = "uint8")
    upper = np.array([20, 255, 255], dtype = "uint8")
    with open(pickle_file, "rb") as f:
        hist = pickle.load(f)
    hist_h = np.sum(hist, axis = 1) # H from HSV
    hist_s = np.sum(hist, axis = 0) # S from HSV
    h_min, h_max = np.nonzero(hist_h)[0][0],  np.nonzero(hist_h)[0][-1]
    s_min, s_max = np.nonzero(hist_s)[0][0],  np.nonzero(hist_s)[0][-1]
    # TODO: implement strictness
    h_min = lower[0] if h_min < lower[0] else h_min 
    h_max = upper[0] if h_max > upper[0] else h_max 
    s_min = lower[1] if h_min < lower[1] else s_min 
    s_max = upper[1] if h_max > upper[1] else s_max 
    hsv_refined_lower = np.array([h_min, s_min, 80], np.uint8)
    hsv_refined_upper = np.array([h_max, s_max, 255], np.uint8)
    with open("hsv_lower", "wb") as f:
        pickle.dump(hsv_refined_lower, f)
    with open("hsv_upper", "wb") as f:
        pickle.dump(hsv_refined_upper, f)
    print()
    print(hsv_refined_lower)
    print(hsv_refined_upper)


def show_webcam(mirror = False, vid_file = 'sample.mp4'):
    cam = cv2.VideoCapture(vid_file)
    valid, im = cam.read()
    if mirror: 
        im = cv2.flip(im, 1)
    im_text = text_on_image( 
            "Please place your hand on the top left\n area of the image  (green rectangle) and let\nthe camera scan it while\nyour show the palm and the back.\nWhen done,\npress the s key to stop scanning.\n\nTry to keep only the skin in the box and \nno other objects.", 
            im)
    top = (0, 0)
    bottom = (int(im_text.shape[1]/6),  int(im_text.shape[0]/6))
    cv2.rectangle(im_text,  top , bottom,  COLOR_GREEN, 4)
    qimshow(im_text, delay = 8)
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    hist_sum = accum_hs_hist(cam, im, top,  bottom, mirror = mirror)

    cv2.destroyAllWindows()


def main():
    # pass vid_file = 0 to read webcam
    show_webcam(mirror=True, vid_file = '../datasets/gesture/test_01_face.mp4')
    extract_skin_range()


if __name__ == '__main__':
    main()
