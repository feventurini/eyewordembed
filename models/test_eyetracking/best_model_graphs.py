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
import chainer.serializers as S
import os
os.chdir('..')

import sys
sys.path.insert(0, '../utilities')
sys.path.insert(0, '.')

import timing
import util
import prepare_dataset as pd

from eyetracking import *
from eyetracking_batch_iter import *

import matplotlib

if __name__ == '__main__':
    gpu = -1
    unit = 100
    batchsize = 1000
    epoch = 1
    window = 1
    out_path = 'test_eyetracking/graphs'

    if gpu >= 0:
        chainer.cuda.get_device_from_id(gpu).use()
        cuda.check_cuda_available()

    vocab, pos2id, train, val, mean, std = pd.load_dataset()
    index2word = {v:k for k,v in vocab.items()}

    n_vocab = len(vocab)
    n_pos = len(pos2id)

    loss_func = F.mean_squared_error

    batch_size = batchsize
    n_units = unit
    window = window

    rule_name = {O.SGD: 'sgd', O.AdaGrad: 'adagrad', O.Adam: 'adam'}
    rules = [O.SGD, O.AdaGrad, O.Adam]

    model_types = ['linreg', 'context_concat', 'multilayer', 'multilayer_context', 'context_sum']
    lr = 0.01
    reg_coeff = 0.001
    wlen = True
    pos = True
    n_pos_units = 50
    outs = ['tanh', 'id']
    n_hidden = 200
    n_layers = 1

    for model_type in model_types:
        for r in rules:
            for out_type in outs:

                if out_type == 'tanh':
                    out = F.tanh
                elif out_type == 'relu':
                    out = F.relu
                elif out_type == 'sigmoid':
                    out = F.sigmoid
                elif out_type == 'id':
                    out = F.identity
                else:
                    raise Exception('Unknown output type: {}'.format(out_type))

                if model_type == 'linreg':
                    model = LinReg(n_vocab, n_units, loss_func, out, wlen=wlen, pos=pos, n_pos=n_pos, n_pos_units=n_pos_units)
                    train_iter = EyeTrackingSerialIterator(train, batch_size, repeat=True, shuffle=True, lens=wlen, pos=pos)
                    val_iter = EyeTrackingSerialIterator(val, batch_size, repeat=False, shuffle=True, lens=wlen, pos=pos)
                elif model_type == 'context_concat':
                    model = LinRegContextConcat(n_vocab, n_units, loss_func, out, window=window, wlen=wlen, pos=pos, n_pos=n_pos, n_pos_units=n_pos_units)
                    train_iter = EyeTrackingWindowIterator(train, window, batch_size, repeat=True, shuffle=True, lens=wlen, pos=pos)
                    val_iter = EyeTrackingWindowIterator(val, window, batch_size, repeat=False, shuffle=True, lens=wlen, pos=pos)
                elif model_type == 'context_sum':
                    model = LinRegContextSum(n_vocab, n_units, loss_func, out, window=window, wlen=wlen, pos=pos, n_pos=n_pos, n_pos_units=n_pos_units)
                    train_iter = EyeTrackingWindowIterator(train, window, batch_size, repeat=True, shuffle=True, lens=wlen, pos=pos)
                    val_iter = EyeTrackingWindowIterator(val, window, batch_size, repeat=False, shuffle=True, lens=wlen, pos=pos)
                elif model_type == 'multilayer':
                    model = Multilayer(n_vocab, n_units, loss_func, out, n_hidden=n_hidden, n_layers=n_layers, wlen=wlen, pos=pos, n_pos=n_pos, n_pos_units=n_pos_units)
                    train_iter = EyeTrackingSerialIterator(train, batch_size, repeat=True, shuffle=True, lens=wlen, pos=pos)
                    val_iter = EyeTrackingSerialIterator(val, batch_size, repeat=False, shuffle=True, lens=wlen, pos=pos)
                elif model_type == 'multilayer_context':
                    model = MultilayerContext(n_vocab, n_units, loss_func, out, n_hidden=n_hidden, n_layers=n_layers, window=1, wlen=wlen, pos=pos, n_pos=n_pos, n_pos_units=n_pos_units)
                    train_iter = EyeTrackingWindowIterator(train, window, batch_size, repeat=True, shuffle=True, lens=wlen, pos=pos)
                    val_iter = EyeTrackingWindowIterator(val, window, batch_size, repeat=False, shuffle=True, lens=wlen, pos=pos)
                else:
                    raise Exception('Unknown model type: {}'.format(model))

                if gpu >= 0:
                    model.to_gpu()

                name = '{}_{}'.format(model_type, out_type)
                optimizer = r(lr)
                optimizer.setup(model)
                l2_reg = chainer.optimizer.WeightDecay(reg_coeff)
                optimizer.add_hook(l2_reg, 'l2')

                updater = training.StandardUpdater(train_iter, optimizer, converter=convert, device=gpu)
                trainer = training.Trainer(updater, (epoch, 'epoch'), out=out_path)

                trainer.extend(extensions.dump_graph('main/loss', out_name=name + '_cg.dot'))
                trainer.run()

