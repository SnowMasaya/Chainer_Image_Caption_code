import unittest
import argparse
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from utils.exclude_gray_image import ExcludeGrayImage
from os import path
APP_ROOT = path.dirname( path.abspath( __file__ ) )


def parse_args(cmdline=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--originalImageFile', '-o', default='',
                        help='set Original Image File')
    parser.add_argument('--gray', '-g', default='',
                        help='set Gray Image File')
    parser.add_argument('--exclude', '-e', default='',
                        help='Exclude Gray Image File')

    if cmdline is not None:
        args = parser.parse_args(cmdline)
    else:
        args = parser.parse_args()

    return args


class TestRead_Data(unittest.TestCase):

    def test_load_image_list(self):
        args = parse_args(["-o", APP_ROOT + "/../Data/index2img.txt",
                           "-g", APP_ROOT + "/../Data/gray_image_list.txt",
                           "-e", APP_ROOT + "/../Data/index2img_exclude.txt"
                           ])
        read_image = ExcludeGrayImage(args.originalImageFile, args.gray,
                                      args.exclude)
        read_image.exclude_gray_image(separator=",")

if __name__ == '__main__':
    unittest.main()
