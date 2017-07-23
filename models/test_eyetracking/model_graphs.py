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
    out_path = 'test_eyetracking/graphs'

    if gpu >= 0:
        chainer.cuda.get_device_from_id(gpu).use()
        cuda.check_cuda_available()


    batch_size = batchsize
    n_units = unit

    r = O.AdaGrad
    lr = 0.01
    reg_coeff = 0.001
    wlen = True
    pos = True
    prev_fix = True
    freq = True
    n_pos_units = 50
    out = F.tanh
    n_hidden = 200

    configs = [(False, 0, 0), (False, 1, 0), (False, 0, 2),
                (True, 0, 0), (True, 1, 0), (True, 0, 2)]

    for config in configs:
        bins, window, n_layers = config

        if bins:
            vocab, pos2id, n_classes, n_participants, train, val, test = pd.load_dataset(bins=True)
        else:
            vocab, pos2id, train, val, test, mean, std = pd.load_dataset()
        
        n_vocab = len(vocab)
        n_pos = len(pos2id)

        if bins:
            loss_func = F.softmax_cross_entropy
        else:
            loss_func = F.mean_squared_error

        if bins:
            model = EyetrackingClassifier(n_vocab, n_units, n_participants, n_classes, loss_func, out, n_hidden=n_hidden, window=window, n_layers=n_layers, wlen=wlen, pos=pos, prev_fix=prev_fix, freq=freq, n_pos=n_pos, n_pos_units=50)
        else:
            model = EyetrackingLinreg(n_vocab, n_units, loss_func, out, n_hidden=n_hidden, window=window, n_layers=n_layers, wlen=wlen, pos=pos, prev_fix=prev_fix, freq=freq, n_pos=n_pos, n_pos_units=50)

        train_iter = EyetrackingBatchIterator(train, window, batch_size, repeat=True, shuffle=True, wlen=wlen, pos=pos, prev_fix=prev_fix, freq=freq, bins=bins)
        val_iter = EyetrackingBatchIterator(val, window, batch_size, repeat=False, shuffle=True, wlen=wlen, pos=pos, prev_fix=prev_fix, freq=freq, bins=bins)


        optimizer = r(lr)
        optimizer.setup(model)
        l2_reg = chainer.optimizer.WeightDecay(reg_coeff)
        optimizer.add_hook(l2_reg, 'l2')

        updater = training.StandardUpdater(train_iter, optimizer, converter=convert, device=gpu)
        trainer = training.Trainer(updater, (epoch, 'epoch'), out='graphs')

        name = '{}_{}window_{}layers'.format(('classifier' if bins else 'linreg'), window, n_layers)
        print(name)
        trainer.extend(extensions.dump_graph('main/loss', out_name=name + '_cg.dot'))

        trainer.run()
