import unittest
import sys
sys.path.insert(1, '../src')
from SkinDetector import * 
import cv2


# limits should be:
# [16 69 80]
# [ 16  69 255]

class TestSkinDetector(unittest.TestCase):
    def setUp(self):
        # create auxilary members
        pass

    def tearDown(self):
        # delete auxilary members
        pass

    def test_hsv_values(self):
        target_h = 16
        target_s = 69 
        # copy workflow from from SkinDetector_run
        sd = SkinDetector()
        # this vid contains only one tone to detect
        # so self.hsv_low[:2] is equal to self.hsv_high[:2]
        vid_file = '../test_vid/test.mp4'
        sd.show_instructions(vid_file)
        sd.extract_skin(vid_file, mirror = False)
        h = sd.hsv_low[0]  
        s = sd.hsv_low[1]  
        self.assertEqual([h, s], [target_h, target_s])

if __name__ == '__main__':
    unittest.main()
