#!/usr/bin/env python
import os
import sys

import numpy
from PIL import Image
import six.moves.cPickle as pickle
list = []
from os import path
APP_ROOT = path.dirname( path.abspath( __file__ ) )


class ComputeMean():

    def __init__(self, dataset, root=".", output_file="mean.npy"):
        self.dataset = dataset
        self.root = root
        self.output = output_file
        self.sum_image = None
        self.count = 0

    def compute_mean_image(self):
        img_list_dir = APP_ROOT + "/../Data/val2014_resize/"
        for line in open(self.dataset):
            filepath = os.path.join(self.root, line.strip().split()[0])
            if len(numpy.asarray(Image.open(img_list_dir + filepath)).shape)==2:
                print (filepath)
            else:
                image = numpy.asarray(Image.open(img_list_dir + filepath)).transpose(2, 0, 1)
                if self.sum_image is None:
                    self.sum_image = numpy.ndarray(image.shape, dtype=numpy.float32)
                    self.sum_image[:] = image
                else:
                    self.sum_image += image
                self.count += 1
                sys.stderr.write('\r{}'.format(self.count))
                sys.stderr.flush()

        sys.stderr.write('\n')

        mean = self.sum_image / self.count
        pickle.dump(mean, open(self.output, 'wb'), -1)
