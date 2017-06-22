#!/usr/bin/env python
"""Sample script of word embedding model.

This code implements skip-gram model and continuous-bow model.
"""
import argparse
import collections

import numpy as np

import chainer
from chainer import cuda
import chainer.functions as F
import chainer.initializers as I
import chainer.links as L
import chainer.optimizers as O
from chainer import reporter
from chainer import training
from chainer.training import extensions

import sys
sys.path.insert(0, '/media/fede/fedeProSD/eyewordembed/utilities')
import timing
import util

class LinReg(chainer.Chain):

    def __init__(self, n_vocab, n_units, loss_func, out):
        super(LinReg, self).__init__()

        with self.init_scope():
            self.embed = L.EmbedID(n_vocab, n_units, initialW=I.GlorotNormal()) # initialW=I.Uniform(1. / n_units))
            self.lin = L.Linear(n_units, 1, initialW=I.GlorotNormal()) # initialW=I.Uniform(1. / n_units))
            self.out = out
            self.loss_func = loss_func

    def __call__(self, w, target):
        e = self.embed(w)
        o = self.out(self.lin(e))
        # for i, j in zip(o, target):
        #     print(i,j)
        # print(self.lin.W.data)
        # input()
        self.loss = self.loss_func(o, target)
        #reporter.report({'loss': loss}, self)
        return self.loss

class LinRegContext(chainer.Chain):

    def __init__(self, n_vocab, n_units, loss_func, out):
        super(LinRegContext, self).__init__()

        with self.init_scope():
            self.embed = L.EmbedID(n_vocab, n_units, initialW=I.Uniform(1. / n_units))
            self.lin = L.Linear(n_units, 1, initialW=I.Uniform(1. / n_units))
            self.out = out
            self.loss_func = loss_func

    def __call__(self, context, target):
        e = self.embed(context)
        h = F.sum(e, axis=1) * (1. / context.shape[1])
        o = self.out(self.lin(h))
        self.loss = self.loss_func(o, target)
        #reporter.report({'loss': loss}, self)
        return self.loss

def convert(batch, device):
    x, targets = batch
    if device >= 0:
        x = cuda.to_gpu(x)
        targets = cuda.to_gpu(targets)
    return x, targets
