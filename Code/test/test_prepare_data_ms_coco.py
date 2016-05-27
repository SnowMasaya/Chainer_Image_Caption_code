import unittest
import argparse
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from utils.prepare_data_ms_coco import Prepare_Data_MS_COCO
from os import path
APP_ROOT = path.dirname( path.abspath( __file__ ) ) + "/../"


def parse_args(cmdline=None):
    parser = argparse.ArgumentParser(description='MS COCO Pre process')
    parser.add_argument('--imageDir', '-i', default='',
                        help='set Image Directory')
    parser.add_argument('--resizeImageDir', '-r', default='',
                        help='set the Resize Image Directory')
    parser.add_argument('--dataDir', '-d', default="",
                        help='anottion data set the Data Directory')
    parser.add_argument('--dataType', '-t', choices=("val2014", "train2014"),
                        default="val2014", help='set the Data type')

    if cmdline is not None:
        args = parser.parse_args(cmdline)
    else:
        args = parser.parse_args()

    return args


class Test_Prepare_Data_MS_COCO(unittest.TestCase):

    def test_load_image_list(self):
        args = parse_args(["-i", APP_ROOT + "Data/val2014",
                           "-r", APP_ROOT + "Data/val2014_resize",
                           "-d", APP_ROOT + "Data",
                           "-t", "val2014"
                           ])
        prepare_data_ms_coco = Prepare_Data_MS_COCO(args.imageDir,
                                                    args.resizeImageDir,
                                                    args.dataDir,
                                                    args.dataType)
        prepare_data_ms_coco.get_name_id()

if __name__ == '__main__':
    unittest.main()
