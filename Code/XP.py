# '!/usr/bin/env python
# -*- coding:utf-8 -*-
# !/usr/bin/python3
from chainer import cuda, Variable
import numpy


class XP:
    __lib = None

    @staticmethod
    def set_library(use_gpu, gpu_id):
        if use_gpu:
            XP.__lib = cuda.cupy
            cuda.get_device(gpu_id).use()
        else:
            XP.__lib = numpy

    @staticmethod
    def __zeros(shape, dtype):
        return Variable(XP.__lib.zeros(shape, dtype=dtype))

    @staticmethod
    def fzeros(shape):
        return XP.__zeros(shape, XP.__lib.float32)

    @staticmethod
    def __array(array, dtype):
        return Variable(XP.__lib.array(array, dtype=dtype))

    @staticmethod
    def iarray(array):
        return XP.__array(array, XP.__lib.int32)

    @staticmethod
    def farray(array):
        return XP.__array(array, XP.__lib.float32)

