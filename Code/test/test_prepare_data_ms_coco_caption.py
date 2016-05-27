import unittest
import argparse
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from utils.prepare_data_ms_coco_caption import Prepare_Data_MS_COCO_Caption
from os import path
APP_ROOT = path.dirname( path.abspath( __file__ ) ) + "/../"

def parse_args(cmdline=None):
    parser = argparse.ArgumentParser(description='MS COCO Pre process')
    parser.add_argument('--imageDir', '-i', default='',
                        help='set Image Directory')
    parser.add_argument('--dataDir', '-d', default="",
                        help='anottion data set the Data Directory')
    parser.add_argument('--dataType', '-t', choices=("val2014", "train2014"),
                        default="val2014", help='set the Data type')

    if cmdline is not None:
        args = parser.parse_args(cmdline)
    else:
        args = parser.parse_args()

    return args


class Test_Prepare_Data_MS_COCO_Caption(unittest.TestCase):

    def test_get_name_id_caption(self):
        val2014 = APP_ROOT + "Data/val2014"
        args = parse_args(["-i", APP_ROOT + "Data/val2014",
                           "-d", APP_ROOT + "Data",
                           "-t", "val2014"
                           ])
        prepare_data_ms_coco_caption = Prepare_Data_MS_COCO_Caption(args.imageDir,
                                                                    args.dataDir,
                                                                    args.dataType)
        prepare_data_ms_coco_caption.get_name_id()

if __name__ == '__main__':
    unittest.main()
