import unittest
import argparse
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from utils.read_image import Read_Image
from os import path
APP_ROOT = path.dirname( path.abspath( __file__ ) )


def parse_args(cmdline=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--imageFile', '-i', default='',
                        help='set Image File')
    parser.add_argument('--mean', '-m', default='',
                        help='set mean File')

    if cmdline is not None:
        args = parser.parse_args(cmdline)
    else:
        args = parser.parse_args()

    return args


class Test_Read_Image(unittest.TestCase):

    def setUp(self):
        self.jpg = "/../Data/val2014/COCO_val2014_000000328299.jpg"
        self.mean_npy = "/../mean.npy"
        self.read_image = ""
        self.args = argparse.ArgumentParser(
            description='Learning convnet from ILSVRC2012 dataset')

    def test_chekc_img_name(self):
        self.args = parse_args(["-i", APP_ROOT + self.jpg,
                                "-m", APP_ROOT + self.mean_npy
                                ])
        self.read_image = Read_Image(self.args.imageFile, self.args.mean)
        self.assertEqual(self.read_image.ImageName, APP_ROOT + self.jpg)
        self.read_image.read_image()

if __name__ == '__main__':
    unittest.main()
