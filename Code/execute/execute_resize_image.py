import unittest
import argparse
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from utils.resize_image import Resize_Image
from os import path
APP_ROOT = path.dirname(path.abspath(__file__))


def parse_args(cmdline=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--imageDir', '-i', default='',
                        help='set Image File')
    parser.add_argument('--resizeDir', '-r', default='',
                        help='set mean File')

    if cmdline is not None:
        args = parser.parse_args(cmdline)
    else:
        args = parser.parse_args()

    return args


class Test_Resize_Image(unittest.TestCase):

    def test_load_image_list(self):
        args = parse_args(["-i", APP_ROOT + "/../Data/val2014",
                           "-r", APP_ROOT + "/../Data/val2014_resize"])
        resize_image = Resize_Image(args.imageDir, args.resizeDir)
        resize_image.resize_image()

if __name__ == '__main__':
    unittest.main()
