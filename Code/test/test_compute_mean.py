import unittest
import argparse
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from utils.compute_mean import ComputeMean
from os import path
APP_ROOT = path.dirname( path.abspath( __file__ ) )


class TestCompute_Mean(unittest.TestCase):

    def test_compute_mean(self):
        img_list = APP_ROOT + "/../Data/img_exclude_gray.txt"
        compute_image = ComputeMean(img_list)
        compute_image.compute_mean_image()

if __name__ == '__main__':
    unittest.main()
