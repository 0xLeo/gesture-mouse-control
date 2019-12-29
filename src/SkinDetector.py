import cv2
import time
import numpy as np
import os


class SkinDetector:
    """
    # Usage example:
    sd = SkinDetector()
    sd.show_instructions(vid_file)
    sd.extract_hsv_model(vid_file)

    sd.stretch_hsv_limits(amount = .1)
    # loop through the video, get current image im and
    sd.apply_mask(im)
    """
    def __init__(self,
            hsv_low = np.array([0, 48, 80], np.uint8),
            hsv_high = np.array([20, 255, 255], np.uint8),
            ycrcb_low = np.array([90, 100, 130], np.uint8),
            ycrcb_high = np.array([230, 120, 180], np.uint8)): 
        # skin detection HSV domain limits
        self.hsv_low = hsv_low
        self.hsv_high = hsv_high
        self.ycrcb_low = ycrcb_low
        self.ycrcb_high = ycrcb_high
        self.timeout = 15
        self._instructions = '\n\n\n\n\n'\
                'Please place your hand in the\n'\
                'green rectangle. When scanning\nstarts,'\
                'scan the palm and the\n'\
                'back for a few seconds.\n'\
                'When ready to start,\npress any key.\n\n'\
                'Try to keep only the skin in the box\n'\
                'and no other objects.' 
        self.mask = None 
        self.accum_hsv_hist = None
 
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
        assert os.path.isfile(vid_file) or\
            isinstance(vid_file,int),\
            "Video file does not exist"
        cap = cv2.VideoCapture(vid_file)
        val, im = cap.read()
        assert val,\
            "Cannot read from camera/ video file"
        while True:
            top = (int(im.shape[0]/7), int(im.shape[1]/7))
            bottom = (int(2 * im.shape[0]/7),
                    int(2 * im.shape[1]/7))
            val, im = cap.read()
            if not val:
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
        cv2.destroyAllWindows()

    def extract_hsv_model(self, vid_file = 0, mirror = True):
        """
        Runs on a video.
        Gets an HSV skin model of the object inside the rectangle
        at top left.
        """
        cap  = cv2.VideoCapture(vid_file)
        val, im = cap.read()
        assert val, "Cannot read from camera"
        
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
            top = (int(im.shape[0]/7), int(im.shape[1]/7)) 
            bottom = (int(2* im.shape[0]/7),
                    int(2 * im.shape[1]/7))
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
        cv2.destroyAllWindows()
        cap.release()

        # remove outliers
        hist_h = np.sum(self.accum_hsv_hist, axis = 1) # H from HSV
        hist_h[hist_h < .05 * hist_h.max()] = .0
        hist_s = np.sum(self.accum_hsv_hist, axis = 0) # S from HSV
        hist_s[hist_s < .05 * hist_s.max()] = .0
        # compute skin HSV limits
        h_min, h_max = np.nonzero(hist_h[1:])[0][0],\
            np.nonzero(hist_h[1:])[0][-1]
        s_min, s_max = np.nonzero(hist_s[1:])[0][0],\
            np.nonzero(hist_s[1:])[0][-1]
        self.hsv_low = np.array([h_min, s_min, 80], np.uint8) 
        self.hsv_high = np.array([h_max, s_max, 255], np.uint8) 


    def stretch_hsv_limits(self, amount = 0.3):
        assert .0 <= amount <= 1.,\
            "Must be a float from 0 to 1."\
            "The higher, the more value it considers as skin."
        h_max, s_max, _ = self.hsv_high
        h_min, s_min, _ = self.hsv_low
        s_max = s_max + amount * (255 - s_max)
        h_max = h_max + .25 * amount * (179 - h_max)
        s_min = s_min - amount * s_min
        h_min = h_min - .25 * amount * h_min
        self.hsv_low = np.array([h_min, s_min, 80], np.uint8)
        self.hsv_high = np.array([h_max, s_max, 255], np.uint8) 


    def apply_hsv_mask(self, im, mirror = True):
        assert len(im.shape) == 3,\
            "This method takes a BGR image and finds the"\
            "skin"
        if mirror:
            im = cv2.flip(im, 1)
        hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        self.mask = cv2.inRange(hsv, self.hsv_low,\
            self.hsv_high)
        hsv = cv2.bitwise_and(hsv, hsv, mask = self.mask)
        im_skin = cv2.bitwise_and(im, im, mask = self.mask) 
        return im_skin


    def backproject(self, bgr, rad = 5):
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        # to get R  (ratio histogram) matrix -> R between 0 and 1
        cv2.normalize(self.accum_hsv_hist, self.accum_hsv_hist, 0, 255, cv2.NORM_MINMAX)
        R = cv2.calcBackProject([hsv],  # image
            [0,1],                  # channel selection
            self.accum_hsv_hist,    # histogram array
            [0,180,0,256],          # channel ranges
            scale = 1)
        # convolve with circular disc
        if rad > 0:
            disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (rad, rad))
            cv2.filter2D(R, -1, disc,R)
        # Otsu's threshold
        _, R_thresh = cv2.threshold(R, 0, 255,  cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        # Make it 3D to AND it with the search image
        R_thresh = cv2.merge((R_thresh, R_thresh, R_thresh))
        return R_thresh 


    def apply_ycrcb_mask(self, im, mirror = True):
        assert len(im.shape) == 3,\
            "This method takes a BGR image and finds the"\
            "skin"
        if mirror:
            im = cv2.flip(im, 1)
        ycrcb = cv2.cvtColor(im, cv2.COLOR_BGR2YCR_CB)
        self.mask = cv2.inRange(ycrcb, self.ycrcb_low,\
            self.ycrcb_high)
        ycrcb = cv2.bitwise_and(ycrcb, ycrcb, mask = self.mask)
        im_skin = cv2.bitwise_and(im, im, mask = self.mask) 
        return im_skin



    def smoothen_mask(self, rad = 9, iterations = 2):
        assert self.mask is not None, \
            "mask processing follows the apply_mask method"
        # TODO: opening followed by closing


    ##
    # @brief 
    #
    # @param im input 3D image
    # @param colour reference colour. We measure the distance from it 
    # @param channels which channels of input image to measure from 
    #
    # @return a numpy array, dark = small distance, white = high
    @classmethod
    def match_colour(self,
            im: np.ndarray,
            colour: tuple,
            channels = [0, 1]) -> np.ndarray:
        """
        Find the difference between the given colour and `im` 
        in a pixel-wise manner.
        The input colour can be in HS domain, e.g. (10, 80).
        `channels` selects which channels of the input image to 
        use, e.g. if the `colour` is in HS domain then use [0,1].
        """
        assert all([0 <= c <= 2 for c in channels]),\
            'channels must be integers from 0 to 2'
        assert len(colour) == len(channels) == 2 or\
                len(colour) == len(channels) == 1,\
                'number of channels must be equal to the dimension of '\
                'reference colour and up to two channels can be used'
        orig_shape = (im.shape[0], im.shape[1])
        if len(channels) == len(colour) == 1:
            c1, ref1 = channels[0], colour
            dist_vec = np.array([np.abs(ref1-im[c1])\
                for im in im.reshape(-1,3)], np.uint8).\
                reshape(orig_shape)
        else:
            c1, c2 = channels
            ref1, ref2 = colour 
            dist_vec = np.array([np.hypot(ref1-im[c1],\
                ref2-im[c2]) for im in im.reshape(-1,3)], np.uint8).\
                reshape(orig_shape)
        return np.bitwise_not(dist_vec).astype(np.uint8)

        
