import unittest
import argparse
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from utils.read_data import Read_Data
from os import path
APP_ROOT = path.dirname( path.abspath( __file__ ) )


def parse_args(cmdline=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_id_file', '-i', default='',
                        help='set Image label file')
    parser.add_argument('--root', '-r', default='',
                        help='set directory')
    parser.add_argument('--caption_file', '-c', default='',
                        help='set caption_file')

    if cmdline is not None:
        args = parser.parse_args(cmdline)
    else:
        args = parser.parse_args()

    return args


class Test_Read_Data(unittest.TestCase):

    def test_load_image_list(self):
        args = parse_args(["-i", APP_ROOT + "/../Data/index2img.txt", "-r", APP_ROOT,
                           "-c", APP_ROOT + "/../Data/index2caption.txt"])
        read_data = Read_Data(args.image_id_file, args.root, args.caption_file)
        read_data.load_image_list()
        read_data.load_caption_list()

if __name__ == '__main__':
    unittest.main()
