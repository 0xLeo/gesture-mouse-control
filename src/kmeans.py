from collections import OrderedDict
import numpy as np
import cv2


##
# @brief Used to count the number of fingers. 
#        Input image must contain the fist and the fingers.
class KMeans:
    '''
    kmeans = KMeans(hsv, 3)
    kmeans.do_kmeans()
    kmeans.find_most_frequent()
    n_fingers = kmeans.count_fingers()
    '''
    def __init__(self, bgr = None, k = 3):
        self.bgr = bgr
        self.k = k
        self.mask = None
        self.segmented = None
        self.centers = None
        self.most_frequent_mask = None
        self.bad_quality = False


    def setup(self, bgr, k):
        self.bgr = bgr
        self.k = k

    ##
    # @brief performs K-means on a 3D image (e.g. BGR).
    #        Writes to self.segmented and self.centers
    #
    # @param X: 3D matrix (image) to perform K-means on
    # @param k: modes of K-means
    #
    # @return: 
    def _do_k_means(self, X, k):
        orig_shape = X.shape
        X = np.float32(X).reshape((-1, 3))

        crit_ = cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER
        max_iter = 10
        epsilon = 1.0
        params = {'K': k,
                'bestLabels': None,
                'criteria': (crit_, max_iter, epsilon),
                'attempts': 10,
                'flags': cv2.KMEANS_RANDOM_CENTERS}
        # labels returns the index of the cluster they belong in
        _, labels, centers = cv2.kmeans(data = X, **params)
        centers = np.uint8(centers)
        # same as for l in flat labels: res.append(center[l])
        res = centers[labels.flatten()]
        res = res.reshape((orig_shape))

        self.segmented = res
        self.centers = centers


    ##
    # @brief applies K-means by calling _do_k_means
    #
    # @return 
    def do_kmeans(self):
        if not (isinstance(self.bgr, np.ndarray)\
                and len(self.bgr.shape) == 3):
            raise TypeError("do_kmeans method needs a 3D image.")
        self._do_k_means(self.bgr, self.k)


    ##
    # @brief performs (optional) morph opening
    #        followed by (optional) morph closing
    #        Writes to self.mask
    #
    # @param rad radius of structuring element
    # @param iters number of opening/ closing iterations
    # @param do_close a flag (bool)
    # @param do_open a flag
    #
    # @return 
    def _smothen_mask(self, iterations = 1,
            do_close = True, do_open = False):
        rad = int(np.sqrt(.01 * self.mask.size)) | 1
        strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                (rad,rad))
        if do_close:
            self.mask = cv2.morphologyEx(self.mask,
                    cv2.MORPH_CLOSE, strel, iterations = 2)
        if do_open:
            self.mask = cv2.morphologyEx(self.mask,
                    cv2.MORPH_OPEN, strel, iterations = 2)
       

    ##
    # @brief finds most frequent colour in segmented image.
    #        Paints it white, the rest black in a mask.
    #        Writes result to self.mask
    #
    # @return 
    def find_most_frequent(self, do_open = False, do_close = True):
        if not (isinstance(self.bgr, np.ndarray)\
                and len(self.bgr.shape) == 3):
            raise TypeError("do_kmeans method needs a 3D image.")
        if self.segmented is None:
            raise ValueError("find_most_frequent needs do_kmeans",
                "to be called prior to it.")

        # map number of pixels of each colour to its frequency
        seg = self.segmented
        npix_to_col = OrderedDict()
        for c in self.centers:
            npix_to_col[np.count_nonzero(seg == c)] = c
        # crop the ROI - the fingers only
        crop = seg[int(seg.shape[0]/18):int(seg.shape[0]/2.2),:]
        # create a mask which is white at the most frequent colour
        max_freq = [k for k in npix_to_col.keys()][0]
        most_freq_col = npix_to_col[max_freq]
        maskBW = np.zeros_like(crop)
        maskBW[crop == most_freq_col] = 255
        self.mask = maskBW[:,:,0]
        if do_close or do_open:
            self._smothen_mask(self.mask, do_close, do_open)


    ##
    # @brief make sure fg (fingers) are white
    #        Writes to self.mask
    #
    # @return 
    def _make_fg_white(self):
        if self.mask is None:
            raise ValueError("_make_fg_white needs a mask."
                "This is found by calling find_most_frequent.")
        if np.count_nonzero(self.mask) > 0.4*self.mask.size:
            self.mask = np.asarray(self.mask, np.int)
            self.mask = np.array(255 - self.mask,np.uint8) 


    ##
    # @brief Given the BW mask (self.mask), count #fingers
    #        Currently supports counting 0, 1, or 2 fingers
    #
    # @param debug
    #
    # @return number of fingers detected
    def count_fingers(self, debug = False):
        if self.mask is None:
            raise ValueError("count_fingers needs a mask."
                "This is found by calling find_most_frequent.")
        self._make_fg_white()
        conts, _ = \
            cv2.findContours(self.mask, cv2.RETR_TREE,\
                cv2.CHAIN_APPROX_SIMPLE)
        if debug:
            img_contours = np.zeros_like(self.mask)
            cv2.drawContours(img_contours, conts, -1, (125,125,70), 3)
            cv2.imshow("", img_contours)
            while (cv2.waitKey() & 0xff != ord('q')): pass
            cv2.destroyAllWindows()

        # look into geometrical properties of contours
        if len(conts) == 0 :
            return 0
        conts = sorted(conts, key = cv2.contourArea, reverse= True)[:2]
        # 0 means zero fingers - just noise
        # if it cannot decide, return 1
        # 1 should be the most common of the 3 gestures
        if len(conts) == 1:
            if cv2.contourArea(conts[0]) < 0.05*self.mask.size:
                return 0
        elif len(conts) == 2:
            if cv2.contourArea(conts[0]) +\
                    cv2.contourArea(conts[1]) < 0.05*self.mask.size:
                return 0
            #TODO: sometimes fingers are combined at the bottom
            # so it thinks they're 1!
            # then 2 contours of similar size => 2 fingers
            if cv2.contourArea(conts[1]) > .5*cv2.contourArea(conts[0]) :
                return 2 
        return 1


         
