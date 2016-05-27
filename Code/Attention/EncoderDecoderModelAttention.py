# '!/usr/bin/env python
# -*- coding:utf-8 -*-
# !/usr/bin/python3

from chainer import cuda, functions, optimizer, \
    optimizers, serializers, link

import utils.generators as gens
from utils.functions import trace, fill_batch
from utils.vocabulary import Vocabulary
from EncoderDecoderAttention import EncoderDecoderAttention
from utils.common_function import CommonFunction
import random
from XP import XP
from sklearn import preprocessing


class EncoderDecoderModelAttention:

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
        self.model            = "ChainerDialogue"
        self.trg_batch        = []
        self.trg_vocab        = []
        self.is_training      = True
        self.generation_limit = 0
        self.encdec           = EncoderDecoderAttention(self.vocab, self.embed, self.hidden,
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

    def __forward_word(self, h):
        first_flag = True
        t = XP.iarray([self.trg_stoi('<s>') for _ in range(self.batch_size)])
        if self.is_training:
            loss = XP.fzeros(())
            for l in range(self.trg_len):
                if first_flag:
                    #h.data = preprocessing.normalize(h.data)
                    self.encdec.h = h
                    y = self.encdec.decode(t)
                    first_flag = False
                else:
                    y = self.encdec.decode(t)
                t = XP.iarray([self.trg_stoi(self.trg_batch[k][l]) for k in range(self.batch_size)])
                loss += functions.softmax_cross_entropy(y, t)
                output = cuda.to_cpu(y.data.argmax(1))
                for k in range(self.batch_size):
                    self.hyp_batch[k].append(self.trg_itos(output[k]))
            print("--------------------loss-----------------")
            print(loss.data)
            return self.hyp_batch, loss

        else:
            while len(self.hyp_batch[0]) < self.generation_limit:
                if first_flag:
                    self.encdec.h = h
                    y = self.encdec.decode(t)
                    first_flag = False
                else:
                    y = self.encdec.decode(t)
                output = cuda.to_cpu(y.data.argmax(1))
                t = XP.iarray(output)
                for k in range(self.batch_size):
                    self.hyp_batch[k].append(self.trg_itos(output[k]))
                if all(self.hyp_batch[k][-1] == '</s>' for k in range(self.batch_size)):
                    break

        return self.hyp_batch

    def forward(self, trg_vocab, is_training, generation_limit):
        batch_size = len(self.trg_batch)
        self.batch_size = batch_size
        self.trg_len = len(self.trg_batch[0]) if self.trg_batch else 0
        self.trg_stoi = trg_vocab.stoi
        self.trg_itos = trg_vocab.itos
        self.trg_vocab = trg_vocab
        self.is_training = is_training
        self.generation_limit = generation_limit
        self.encdec.reset(batch_size)

        self.hyp_batch = [[] for _ in range(batch_size)]

        h = self.__forward_img()
        return self.__forward_word(h)

    def train(self):
        trace('making vocabularies ...')
        trg_vocab = Vocabulary.new(gens.word_list(self.target), self.vocab)

        trace('making model ...')

        for epoch in range(self.epoch):
            trace('epoch %d/%d: ' % (epoch + 1, self.epoch))
            trained = 0
            opt = optimizers.AdaGrad(lr=0.01)
            opt.setup(self.encdec)
            opt.add_hook(optimizer.GradientClipping(5))
            gen1 = gens.word_list(self.target)
            gen = gens.batch(gen1, self.minibatch)

            random_number = random.randint(0, self.minibatch - 1)
            for trg_batch in gen:
                self.trg_batch = fill_batch(trg_batch)
                if len(self.trg_batch) != self.minibatch:
                    break
                hyp_batch, loss = self.forward(trg_vocab, self.use_gpu, self.gpu_id)
                loss.backward()
                opt.update()
                K = len(self.trg_batch)

                if trained == 0:
                    self.print_out(random_number, epoch, trained, hyp_batch)

                trained += K

        trace('saving model ...')
        prefix = self.model
        trg_vocab.save(prefix + '.trgvocab')
        self.encdec.save_spec(prefix + '.spec')
        serializers.save_hdf5(prefix + '.weights', self.encdec)

        trace('finished.')

    def test(self):
        trace('loading model ...')
        trg_vocab = Vocabulary.load(self.model + '.trgvocab')
        self.encdec = EncoderDecoderAttention.load_spec(self.model + '.spec')
        serializers.load_hdf5(self.model + '.weights', self.encdec)

        trace('generating translation ...')
        generated = 0

        trace('sample %8d - %8d ...' % (generated + 1, generated))
        hyp_batch = self.forward(trg_vocab, False, self.generation_limit)

        source_cuont = 0
        with open(self.target, 'w') as fp:
            for hyp in hyp_batch:
                hyp.append('</s>')
                hyp = hyp[: hyp.index('</s>')]
                print('hyp : ' + ''.join(hyp))
                fp.write(' '.join(hyp))
                source_cuont = source_cuont + 1

        trace('finished.')

    def print_out(self, K, i_epoch, trained, hyp_batch):

        # trace('epoch %3d/%3d, sample %8d' % (i_epoch + 1, self.epoch, trained + K + 1))
        trace('epoch %3d/%3d, sample %8d' % (i_epoch + 1, self.epoch, trained + 1))
        trace('  trg = ' + ' '.join([x if x != '</s>' else '*' for x in self.trg_batch[K]]))
        trace('  hyp = ' + ' '.join([x if x != '</s>' else '*' for x in hyp_batch[K]]))

    def copy_model(self, src, dst, dec_flag=False):
        print("start copy")
        for child in src.children():
            if child.name not in dst.__dict__: continue
            dst_child = dst[child.name]
            if type(child) != type(dst_child): continue
            if isinstance(child, link.Chain):
                self.copy_model(child, dst_child)
            if isinstance(child, link.Link):
                match = True
                for a, b in zip(child.namedparams(), dst_child.namedparams()):
                    if a[0] != b[0]:
                        match = False
                        break
                    if a[1].data.shape != b[1].data.shape:
                        match = False
                        break
                if not match:
                    print('Ignore %s because of parameter mismatch' % child.name)
                    continue
                for a, b in zip(child.namedparams(), dst_child.namedparams()):
                    b[1].data = a[1].data
                    print('Copy %s' % child.name)
