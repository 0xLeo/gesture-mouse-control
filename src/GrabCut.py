import numpy as np
import cv2

class GrabCut:
    '''
    # workflow:
    grab = GrabCut(im)
    grab.create_circular_mask((411, 813), 50)
    grab.apply_grab_cut_mask()
    '''
    def __init__(self, im):
        assert len(im.shape) == 3,\
                "Wrong input. Input is a BGR image"
        self.im = im
        self.mask = np.zeros(self.im.shape[:2],np.uint8)
        self.mask[:,:] = cv2.GC_PR_BGD
        self.output = None

    def create_circular_mask(self, center, rad):
        assert type(center) is tuple, \
            "parameter center must be a tuple"
        for r in range(center[1] - rad, center[1] + rad):
            for c in range(center[0] - rad, center[0] + rad):
                # TODO: is in image? check for edges
                try:
                    if (r - center[1])**2 + (c - center[0])**2 <= \
                            rad**2:
                        self.mask[c, r] = cv2.GC_PR_FGD
                except:
                    pass

    def apply_grab_cut_mask(self, iterations = 2):
        # don't need it if we use mask
        rect = (0, 0, 0, 0)
        bgdModel = np.zeros((1,65),np.float64)
        fgdModel = np.zeros((1,65),np.float64)
        # TODO: how's complexity affected by no iterations?
        cv2.grabCut(self.im,
                self.mask,
                rect,
                bgdModel,
                fgdModel,
                iterations,
                cv2.GC_INIT_WITH_MASK)
        mask2 = np.where((self.mask == 2) | (self.mask == 0), 0, 1).astype('uint8')
        self.output = self.im * mask2[:, :, np.newaxis]


    @staticmethod
    def qimshow(im, wname = 'display', timeout = 5):
        cv2.imshow(wname, im)
        cv2.waitKey(timeout * 1000)
        cv2.destroyAllWindows()



