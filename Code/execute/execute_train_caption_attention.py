import unittest
import argparse
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from train_caption_attention_cpu import TrainCaptionAttention
# Use GPU
#from train_caption_attention import TrainCaptionAttention


def parse_args(cmdline=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_gpu', '-gu', default='',
                        help='set Image File')
    parser.add_argument('--gpu_id', '-gi', default='',
                        help='set Image File')

    if cmdline is not None:
        args = parser.parse_args(cmdline)
    else:
        args = parser.parse_args()

    return args


class Test_Train_Caption(unittest.TestCase):

    def test_load_image_list(self):
        #args = parse_args(["-gu", True, "-gi", 0])
        train_caption = TrainCaptionAttention()
        # USE GPU
        #train_caption.train(True, 0)
        train_caption.train(False, -1)

if __name__ == '__main__':
    unittest.main()
