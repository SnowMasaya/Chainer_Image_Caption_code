# '!/usr/bin/env python
# -*- coding:utf-8 -*-
# !/usr/bin/python3

from chainer import cuda, functions, optimizer, \
    optimizers, serializers, link

import utils.generators as gens
from utils.functions import trace, fill_batch
from utils.vocabulary import Vocabulary
from EncoderDecoder import EncoderDecoder
from utils.common_function import CommonFunction
import random
from XP import XP
from sklearn import preprocessing


class EncoderDecoderModel:

    def __init__(self, parameter_dict):
        self.parameter_dict   = parameter_dict
        self.id2image         = parameter_dict["x"]
        self.first_word       = parameter_dict["first_word"]
        self.target           = parameter_dict["target"]
        self.vocab            = parameter_dict["vocab"]
        self.embed            = parameter_dict["embed"]
        self.hidden           = parameter_dict["hidden"]
        self.epoch            = parameter_dict["epoch"]
        self.minibatch        = parameter_dict["minibatch"]
        self.generation_limit = parameter_dict["generation_limit"]
        self.use_gpu          = parameter_dict["use_gpu"]
        self.gpu_id           = parameter_dict["gpu_id"]
        self.choose_model     = parameter_dict["choose_model"]
        self.common_function  = CommonFunction()
        self.model            = "ChainerImageCaption"
        self.trg_batch        = []
        self.trg_vocab        = []
        self.is_training      = True
        self.generation_limit = 0
        self.encdec           = EncoderDecoder(self.vocab, self.embed, self.hidden,
                                               self.choose_model,
                                               self.use_gpu, self.gpu_id)
        if self.use_gpu:
            self.encdec.to_gpu()
        self.__set_gpu()

    def __set_gpu(self):
        XP.set_library(self.use_gpu, self.gpu_id)

    def __forward_img(self):
        x = XP.farray(self.id2image.data)
        return self.encdec.encode(x)

    def __forward_word(self):
        t = XP.iarray([self.trg_stoi('<s>') for _ in range(self.batch_size)])
        hyp_batch = [[] for _ in range(self.batch_size)]
        if self.is_training:
            loss = XP.fzeros(())
            for l in range(self.trg_len):
                y = self.encdec.decode(t)
                t = XP.iarray([self.trg_stoi(self.trg_batch[k][l]) for k in range(self.batch_size)])
                loss += functions.softmax_cross_entropy(y, t)
                output = cuda.to_cpu(y.data.argmax(1))
                for k in range(self.batch_size):
                    hyp_batch[k].append(self.trg_itos(output[k]))
            return loss, hyp_batch

    def forward(self, is_training, generation_limit, epoch):
        self.is_training = is_training
        self.generation_limit = generation_limit

        for trg_batch in self.gen:
            self.trg_batch = fill_batch(trg_batch)
            if len(self.trg_batch) != self.minibatch:
                break
            self.batch_size = len(self.trg_batch)
            self.trg_len = len(self.trg_batch[0]) if self.trg_batch else 0
            self.encdec.clear(self.batch_size)
            self.__forward_img()
            self.encdec.reset(self.batch_size)
            loss, hyp_batch = self.__forward_word()
            loss.backward()
            self.opt.update()
            K = len(self.trg_batch) - 2
            self.print_out(K, hyp_batch, epoch)

    def train(self):
        trace('making vocabularies ...')
        self.trg_vocab = Vocabulary.new(gens.word_list(self.target), self.vocab)
        self.trg_stoi = self.trg_vocab.stoi
        self.trg_itos = self.trg_vocab.itos

        trace('making model ...')

        for epoch in range(self.epoch):
            trace('epoch %d/%d: ' % (epoch + 1, self.epoch))
            self.opt = optimizers.AdaGrad(lr=0.01)
            self.opt.setup(self.encdec)
            self.opt.add_hook(optimizer.GradientClipping(5))
            gen1 = gens.word_list(self.target)
            self.gen = gens.batch(gen1, self.minibatch)
            self.forward(True, 0, epoch)


        trace('saving model ...')
        prefix = self.model
        self.trg_vocab.save("model/" + prefix + '.trgvocab')
        self.encdec.save_spec("model/" + prefix + '.spec')
        serializers.save_hdf5("model/" + prefix + '.weights', self.encdec)

        trace('finished.')


    def print_out(self, K, hyp_batch, epoch):

        trace('epoch %3d/%3d, sample %8d' % (epoch, self.epoch, K + 1))
        # trace('epoch %3d/%3d, sample %8d' % (i_epoch + 1, self.epoch, trained + 1))
        trace('  trg = ' + ' '.join([x if x != '</s>' else '*' for x in self.trg_batch[K]]))
        trace('  hyp = ' + ' '.join([x if x != '</s>' else '*' for x in hyp_batch[K]]))
