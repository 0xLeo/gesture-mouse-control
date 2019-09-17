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

    def test_create_skin_detector(self):
        """
        Require skin detector to have some hsv_low and hsv_high
        limits, ideally set to either None or some default value.
        """
        test1_passed = False
        test2_passed = False
        test3_passed = False
        sd = SkinDetector()
        if hasattr(sd, 'hsv_low') and hasattr(sd, 'hsv_high'):
            test1_passed = True 
            if sd.hsv_low is not None and\
                    sd.hsv_high is not None:
                test2_passed = True
            if isinstance(sd.hsv_high, np.ndarray) and\
                    isinstance(sd.hsv_low, np.ndarray):
                if len(sd.hsv_high) == len(sd.hsv_low) == 3 and\
                        all(0 <= h < 256 for h in sd.hsv_low) and\
                        all(0 <= h < 256 for h in sd.hsv_high):
                            test3_passed = True
        self.assertEqual(test1_passed, True)
        self.assertEqual(test2_passed, True)
        self.assertEqual(test3_passed, True)

if __name__ == '__main__':
    unittest.main()
