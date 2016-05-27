# '!/usr/bin/env python
# -*- coding:utf-8 -*-
# !/usr/bin/python3

from chainer import Variable
import numpy as np


class CommonFunction():

    def my_zeros(self, shape, dtype):
        return Variable(np.zeros(shape, dtype=dtype))

    def my_array(self, array, dtype):
        return Variable(np.array(array, dtype=dtype))
