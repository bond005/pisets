import os
import sys
import unittest


try:
    from utils.utils import time_to_str
except:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from utils.utils import time_to_str


class TestUtils(unittest.TestCase):
    def test_time_to_str_pos01(self):
        t = 30.56789
        true_str = '00:00:30.568'
        self.assertTrue(true_str, time_to_str(t))

    def test_time_to_str_pos02(self):
        t = 150.56789
        true_str = '00:02:30.568'
        self.assertTrue(true_str, time_to_str(t))

    def test_time_to_str_pos03(self):
        t = 750.56789
        true_str = '00:12:30.568'
        self.assertTrue(true_str, time_to_str(t))

    def test_time_to_str_pos04(self):
        t = 4110.56789
        true_str = '01:08:30.568'
        self.assertTrue(true_str, time_to_str(t))

    def test_time_to_str_pos05(self):
        t = 214110.56721
        true_str = '59:28:30.567'
        self.assertTrue(true_str, time_to_str(t))

    def test_time_to_str_neg01(self):
        t = -3.4
        with self.assertRaises(ValueError):
            _ = time_to_str(t)

    def test_time_to_str_neg02(self):
        t = '30.56789'
        with self.assertRaises(ValueError):
            _ = time_to_str(t)


if __name__ == '__main__':
    unittest.main(verbosity=2)
