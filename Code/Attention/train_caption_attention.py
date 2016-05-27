# '!/usr/bin/env python
# -*- coding:utf-8 -*-
# !/usr/bin/python3

from utils.read_data import Read_Data
from os import path
APP_ROOT = path.dirname( path.abspath( __file__ ) )

import numpy as np

import chainer
from chainer import cuda
from PIL import Image

from EncoderDecoderModelAttention import EncoderDecoderModelAttention
import utils.generators as gens
from utils.vocabulary import Vocabulary
import six.moves.cPickle as pickle


class TrainCaptionAttention():

    def __init__(self):
        self.parameter_dict = {}
        train_path = APP_ROOT + "/Data/"

        self.parameter_dict["id2image"]         = train_path + "index2img_exclude.txt"
        self.parameter_dict["id2caption"]       = train_path + "index2caption.txt"
        self.parameter_dict["target"]           = train_path + "index2caption.txt"
        self.parameter_dict["vocab"]            = 5000
        self.parameter_dict["embed"]            = 300
        self.parameter_dict["hidden"]           = 500
        self.parameter_dict["epoch"]            = 20
        self.parameter_dict["minibatch"]        = 64
        self.parameter_dict["generation_limit"] = 256
        self.parameter_dict["use_gpu"]          = True 
        self.parameter_dict["gpu_id"]           = 0
        self.parameter_dict["choose_model"] = "Alex_Model"

        if self.parameter_dict["choose_model"] == "Alex_Model":
            self.insize = 224
        if self.parameter_dict["choose_model"] == "AlexBn_Model":
            self.insize = 227

        mean_image = pickle.load(open("mean.npy", 'rb'))

        cropwidth = 256 - self.insize
        self.start = cropwidth // 2
        self.stop = self.start + self.insize
        self.mean_image = mean_image[:, self.start:self.stop, self.start:self.stop].copy()

        self.x_batch = np.ndarray((self.parameter_dict["minibatch"], 3,
                                   self.insize, self.insize), dtype=np.float32)
        self.y_batch = np.ndarray((self.parameter_dict["minibatch"]),
                                  dtype=np.int32)

        self.trg_vocab = Vocabulary.new(gens.word_list(self.parameter_dict["target"]), self.parameter_dict["vocab"])
        self.read_data = Read_Data(self.parameter_dict["id2image"],
                                   "Data/val2014_resize",
                                   self.parameter_dict["id2caption"])
        self.read_data.load_image_list()
        self.read_data.load_caption_list()

    def train(self, use_gpu, gpu_id):
        if use_gpu:
            cuda.check_cuda_available()
        xp = cuda.cupy if gpu_id >= 0 and use_gpu == True else np
        batch_count = 0
        for k, v in self.read_data.total_words_ids.items():
            if k in self.read_data.images_ids:
                image = np.asarray(Image.open(APP_ROOT + "/" + self.read_data.images_ids[k])).transpose(2, 0, 1)[::-1]
                image = image[:, self.start:self.stop, self.start:self.stop].astype(np.float32)
                image -= self.mean_image

                self.x_batch[batch_count] = image
                self.y_batch[batch_count] = self.trg_vocab.stoi(self.read_data.total_words_ids[k].split()[0])

                if batch_count < self.parameter_dict["minibatch"]:
                    x_data = xp.asarray(self.x_batch)
                    y_data = xp.asarray(self.y_batch)

                    x = chainer.Variable(x_data, volatile=True)
                    t = chainer.Variable(y_data, volatile=True)
                    self.parameter_dict["x"] = x
                    self.parameter_dict["first_word"] = t
                    encoderDecoderModel = EncoderDecoderModelAttention(self.parameter_dict)
                    encoderDecoderModel.train()
                    batch_count = 0
                batch_count = batch_count + 1
