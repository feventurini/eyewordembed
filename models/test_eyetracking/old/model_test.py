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
    epoch = 50
    window = 1
    reg_coeff = 0.001
    out_path = 'test_eyetracking/result/initial_exploration'

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

    model_types = ['linreg', 'context_sum', 'context_concat']
    wlens = [True, False]
    poss = [True, True, True, False]
    n_pos_unitss = [50, 100, 200, None]
    outs = ['tanh', 'relu', 'id', 'sigmoid']

    for model_type in model_types:
        for out_type in outs:
            for wlen in wlens:
                for n_pos_units, pos in zip(n_pos_unitss, poss):

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
                    elif model_type == 'context_sum':
                        model = LinRegContextSum(n_vocab, n_units, loss_func, out, window=window, wlen=wlen, pos=pos, n_pos=n_pos, n_pos_units=n_pos_units)
                        train_iter = EyeTrackingWindowIterator(train, window, batch_size, repeat=True, shuffle=True, lens=wlen, pos=pos)
                        val_iter = EyeTrackingWindowIterator(val, window, batch_size, repeat=False, shuffle=True, lens=wlen, pos=pos)
                    elif model_type == 'context_concat':
                        model = LinRegContextConcat(n_vocab, n_units, loss_func, out, window=window, wlen=wlen, pos=pos, n_pos=n_pos, n_pos_units=n_pos_units)
                        train_iter = EyeTrackingWindowIterator(train, window, batch_size, repeat=True, shuffle=True, lens=wlen, pos=pos)
                        val_iter = EyeTrackingWindowIterator(val, window, batch_size, repeat=False, shuffle=True, lens=wlen, pos=pos)
                    elif model_type == 'baseline':
                        model = Baseline(0.0, loss_func)
                        train_iter = EyeTrackingSerialIterator(train, batch_size, repeat=True, shuffle=True)
                        val_iter = EyeTrackingSerialIterator(val, batch_size, repeat=False, shuffle=True)
                    else:
                        raise Exception('Unknown model type: {}'.format(model))

                    if gpu >= 0:
                        model.to_gpu()

                    name = '{}_len{}_pos{}_posUnits{}'.format(out_type, wlen, pos, n_pos_units)
                    print('{}_{}'.format(model_type, name))

                    optimizer = O.Adam()
                    optimizer.setup(model)
                    # l2_reg = chainer.optimizer.WeightDecay(reg_coeff)
                    # optimizer.add_hook(l2_reg, 'l2')

                    updater = training.StandardUpdater(train_iter, optimizer, converter=convert, device=gpu)
                    trainer = training.Trainer(updater, (epoch, 'epoch'), out=out_path + os.sep + model_type)

                    trainer.extend(extensions.Evaluator(val_iter, model, converter=convert, device=gpu))

                    trainer.extend(extensions.LogReport(log_name='{}.log'.format(name)))
                    trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'validation/main/loss']))

                    plot_rep = extensions.PlotReport(['main/loss', 'validation/main/loss'], file_name=name + '.pdf')
                    trainer.extend(plot_rep)

                    trainer.extend(extensions.ProgressBar())
                    trainer.run()


