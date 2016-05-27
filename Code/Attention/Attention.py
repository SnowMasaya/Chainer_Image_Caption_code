# '!/usr/bin/env python
# -*- coding:utf-8 -*-
# !/usr/bin/python3

from chainer import Chain, functions, links
from XP import XP


class Attention(Chain):
    def __init__(self, hidden_size ,resize_images):
        super(Attention, self).__init__(
            aw=links.Linear(resize_images, resize_images),
            pw=links.Linear(hidden_size, hidden_size),
            we=links.Linear(hidden_size, 1),
        )
        self.hidden_size = hidden_size

    def __call__(self, a_list, b_list, c_list, p):
        batch_size = p.data.shape[0]
        e_list= []
        sum_e = XP.fzeros((batch_size, 1))
        for a, b, c in zip(a_list, b_list, c_list):
            total_attention = [a, b, c]
            w = functions.tanh(self.aw(XP.farray(a)) + self.pw(p))
            e = functions.exp(self.we(w))
            e_list.append(e)
            sum_e += e
        ZEROS = XP.fzeros((batch_size, self.hidden_size))
        aa = ZEROS
        bb = ZEROS
        cc = ZEROS
        for a, b, c, e in zip(a_list, b_list, c_list, e_list):
            e /= sum_e
            aa += functions.reshape(functions.basic_matmul(a, e), (batch_size, self.hidden_size))
            bb += functions.reshape(functions.basic_matmul(b, e), (batch_size, self.hidden_size))
            cc += functions.reshape(functions.basic_matmul(c, e), (batch_size, self.hidden_size))
        return aa, bb, cc
