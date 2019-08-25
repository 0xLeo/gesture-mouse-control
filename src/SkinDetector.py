import cv2
import time
import numpy as np


class SkinDetector:

    """
    # Usage example:
    sd = SkinDetector()
    sd.show_instructions(vid_file)
    sd.extract_skin(vid_file)

    sd.stretch_hsv_limits(amount = .1)
    sd.apply_mask(vid_file)
    """
    def __init__(self,
            hsv_low = np.array([0, 48, 80], np.uint8),
            hsv_high = np.array([20, 255, 255], np.uint8)):
        # skin detection HSV domain limits
        self.hsv_low = hsv_low
        self.hsv_high = hsv_high
        self.timeout = 15
        self.accum_hsv_hist = np.zeros((180, 256), np.float32)
        self._instructions = '\n\n\n\n\n'\
                'Please place your hand in the\n'\
                'green rectangle. When scanning\nstarts,'\
                'scan the palm and the\n'\
                'back for a few seconds.\n'\
                'When ready to start,\npress any key.\n\n'\
                'Try to keep only the skin in the box\n'\
                'and no other objects.' 
        # 0 -> webcam
        self.vid_file = 0
        self.mask = []
 

    @staticmethod
    def text_on_image(text, im, y0 = 30):
        white = (255, 255, 255)
        # y0 is the text position
        y0,  dy = y0, 30
        for i, line in enumerate(text.split('\n')):
            y = y0 + i*dy
            cv2.putText(im,
                line,
                (50, y ),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                color = white,
                thickness = 2)
        return im 


    def show_instructions(self, vid_file = 0, mirror = True):
        cap = cv2.VideoCapture(vid_file)
        val, im = cap.read()
        assert val,\
            "Cannot read from camera/ video file"
        while True:
            top = (0, 0)
            bottom = (int(im.shape[0]/7),
                    int(im.shape[1]/7))
            val, im = cap.read()
            if val:
                break
            if mirror:
                im = cv2.flip(im, 1)
            # text and rectangle
            self.text_on_image(self._instructions, im)
            cv2.rectangle(im, top, bottom, (40, 255, 0), 4)
            cv2.imshow("skin sample", im)
            k = cv2.waitKey(20)
            if k != -1:
                break


    def extract_skin(self, vid_file = 0, mirror = True):
        """
        Gets an HSV skin model of the object inside the rectangle
        at top left.
        """
        cap  = cv2.VideoCapture(vid_file)
        val, im = cap.read()
        
        self.accum_hsv_hist = np.zeros((180,256), np.float32) 
        time_start = time.time()
        while val and time.time() - time_start < self.timeout:
            val, im = cap.read()
            if not val:
                break
            if mirror:
                im = cv2.flip(im, 1)
            #im = cv2.GaussianBlur(im, (13, 13), 8.25,\
                    #im.copy(), 8.25)
            hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
            top = (0, 0)
            bottom = (int(im.shape[0]/7),
                    int(im.shape[1]/7))
            # this is where we get the skin model from
            hsv_cropped = hsv[top[1]:bottom[1], top[0]:bottom[0]] 
            hs_hist = cv2.calcHist([hsv_cropped], [0, 1], None,
                    [180, 256], [0, 180, 0, 256])
            # histogram - which pixels are "good" for skin model
            self.accum_hsv_hist += hs_hist
            self.accum_hsv_hist = self.accum_hsv_hist /\
                    np.sum(hs_hist)
            cv2.rectangle(im, top, bottom, (40, 255, 0), 4)
            cv2.imshow("skin sample", im)
            k = cv2.waitKey(20)
            if k != -1:
                break
        # remove outliers
        hist_h = np.sum(self.accum_hsv_hist, axis = 1) # H from HSV
        hist_h[hist_h < .1 * hist_h.max()] = .0
        hist_s = np.sum(self.accum_hsv_hist, axis = 0) # S from HSV
        hist_s[hist_s < .1 * hist_s.max()] = .0
        # compute skin HSV limits
        h_min, h_max = np.nonzero(hist_h[1:])[0][0],\
                np.nonzero(hist_h[1:])[0][-1]
        s_min, s_max = np.nonzero(hist_s[1:])[0][0],\
                np.nonzero(hist_s[1:])[0][-1]
        self.hsv_low = np.array([h_min, s_min, 80], np.uint8) 
        self.hsv_high = np.array([h_max, s_max, 255], np.uint8) 


    def stretch_hsv_limits(self, amount = 0.3):
        assert .0 <= amount <= 1.,\
            "That's how I designed it!"
        h_max, s_max, _ = self.hsv_high
        h_min, s_min, _ = self.hsv_low
        s_max = s_max + amount * (255 - s_max)
        h_max = h_max + .25 * amount * (179 - h_max)
        s_min = s_min - amount * s_min
        h_min = h_min - .25 * amount * h_min
        self.hsv_low = np.array([h_min, s_min, 80], np.uint8)
        self.hsv_high = np.array([h_max, s_max, 255], np.uint8) 


    def apply_mask(self, vid_file = 0, mirror = True):
        cap  = cv2.VideoCapture(vid_file)
        val = True
        val, im = cap.read()
        while val:
            val, im = cap.read()
            if mirror:
                im = cv2.flip(im, 1)
            hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
            self.mask = cv2.inRange(hsv, self.hsv_low,\
                    self.hsv_high)
            hsv = cv2.bitwise_and(hsv, hsv, mask = self.mask)
            im_skin = cv2.bitwise_and(im, im, mask= self.mask) 
            cv2.imshow("skin sample", im_skin)
            k = cv2.waitKey(10)
            if k != -1:
                break
        cv2.destroyAllWindows()
