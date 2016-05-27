import numpy as np
from PIL import Image
import random
import six.moves.cPickle as pickle
import sys
import os
from ImageCode.alex_rnnlm import Alex_RNNLM
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
hidden_size = 500
model = Alex_RNNLM(hidden_size)


class Read_Image():

    def __init__(self, imageName, mean):
        self.ImageName = imageName
        self.image = np.asarray([])
        self.cropwidth = 256 - model.insize
        self.model = model
        self.mean_image = pickle.load(open(mean, 'rb'))
        self.read_image_data = np.array([])

    def read_image(self, center=False, flip=False):
        if len(np.asarray(Image.open(self.ImageName)).shape) == 2:
            return
        self.read_image_data = np.asarray(
            Image.open(self.ImageName)).transpose(2, 0, 1)
        if center:
            top = left = self.cropwidth / 2
        else:
            top = random.randint(0, self.cropwidth - 1)
            left = random.randint(0, self.cropwidth - 1)
        bottom = self.model.insize + top
        right = self.model.insize + left

        self.read_image_data = \
            self.read_image_data[:, top:bottom, left:right].astype(np.float32)
        self.read_image_data -= self.mean_image[:, top:bottom, left:right]
        self.read_image_data /= 255
        if flip and random.randint(0, 1) == 0:
            return self.read_image_data[:, :, ::-1]
        else:
            return self.read_image_data
        Image.close(self.ImageName)
