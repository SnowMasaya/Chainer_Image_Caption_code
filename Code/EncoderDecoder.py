# '!/usr/bin/env python
# -*- coding:utf-8 -*-
# !/usr/bin/python3

from chainer import Chain, cuda
from ImageCode.alex_rnnlm import Alex_RNNLM
from ImageCode.alexbn_rnnlm import AlexBn_RNNLM
from Decoder import Decoder
from utils.common_function import CommonFunction
from XP import XP


class EncoderDecoder(Chain):

    def __init__(self, vocab_size, embed_size, hidden_size, choose_model,
                 use_gpu, gpu_id):
        # gpu Setting
        model = Alex_RNNLM(hidden_size)
        if choose_model == "Alex_RNNLM":
            model = Alex_RNNLM(hidden_size)
        if choose_model == "AlexBn_RNNLM":
            model = AlexBn_RNNLM(hidden_size)

        if use_gpu:
            cuda.get_device(gpu_id).use()
            model.to_gpu()
        # Setting Model
        super(EncoderDecoder, self).__init__(
            enc=model,
            dec=Decoder(vocab_size, embed_size, hidden_size),
        )
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.common_function = CommonFunction()
        self.use_gpu = use_gpu
        self.gpu_id = gpu_id
        self.choose_model = choose_model
        self.__set_gpu()

    def __set_gpu(self):
        XP.set_library(self.use_gpu, self.gpu_id)
    
    def clear(self, batch_size):
        self.loss = None
        self.accuracy = None
        self.h = XP.fzeros((batch_size, self.hidden_size))

    def reset(self, batch_size):
        self.zerograds()
        self.c = XP.fzeros((batch_size, self.hidden_size))

    def encode(self, x):
        self.h = self.enc(x)

    def decode(self, y):
        y, self.c, self.h = self.dec(y, self.c, self.h)
        return y

    def save_spec(self, filename):
        with open(filename, 'w') as fp:
            fp.write(str(self.vocab_size))
            fp.write(str(self.embed_size))
            fp.write(str(self.hidden_size))
            fp.write(str(self.choose_model))
            fp.write(str(self.use_gpu))
            fp.write(str(self.gpu_id))

    @staticmethod
    def load_spec(filename):
        with open(filename) as fp:
            vocab_size = int(next(fp))
            embed_size = int(next(fp))
            hidden_size = int(next(fp))
            choose_model = int(next(fp))
            use_gpu = int(next(fp))
            gpu_id = int(next(fp))
            return EncoderDecoder(vocab_size, embed_size, hidden_size,
                                  choose_model, use_gpu, gpu_id)
