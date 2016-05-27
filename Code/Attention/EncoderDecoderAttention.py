# '!/usr/bin/env python
# -*- coding:utf-8 -*-
# !/usr/bin/python3

from chainer import Chain, cuda, links
from ImageCode.alex_rnnlm import Alex_RNNLM
from ImageCode.alexbn_rnnlm import AlexBn_RNNLM
from Decoder import Decoder
from Attention import Attention
from utils.common_function import CommonFunction
from XP import XP
IM_SIZE = 224
RESIZE_IM_SIZE = 24


class EncoderDecoderAttention(Chain):

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
        super(EncoderDecoderAttention, self).__init__(
            enc=model,
            im1 = links.Linear(IM_SIZE, RESIZE_IM_SIZE),
            im2 = links.Linear(IM_SIZE, RESIZE_IM_SIZE),
            im3 = links.Linear(IM_SIZE, RESIZE_IM_SIZE),
            att=Attention(hidden_size, RESIZE_IM_SIZE),
            outay = links.Linear(hidden_size, hidden_size),
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

    def reset(self, batch_size):
        self.zerograds()
        self.x_list = []
        self.c = XP.fzeros((batch_size, self.hidden_size))
        self.h = XP.fzeros((batch_size, self.hidden_size))

    def encode(self, x):
        self.x_list = x
        self.a_list = []
        self.b_list = []
        self.c_list = []
        for x_data in self.x_list.data:
            a = self.im1(XP.farray(x_data[0]))
            b = self.im2(XP.farray(x_data[1]))
            c = self.im3(XP.farray(x_data[2]))
            self.a_list.append(a.data)
            self.b_list.append(b.data)
            self.c_list.append(c.data)
        self.h = self.enc(x)
        aa, bb, cc = self.att(self.a_list, self.b_list, self.c_list, self.h)
        y = self.outay(self.h + aa + bb + cc)
        return y

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

