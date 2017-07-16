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
    epoch = 20
    window = 1
    out_path = 'test_eyetracking/result/'

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

    rule_name = {O.AdaGrad: 'adagrad', O.Adam: 'adam'}
    rules = [O.AdaGrad, O.Adam]

    model_types = ['linreg', 'context_concat', 'multilayer', 'multilayer_context']
    lrs = [0.1, 0.01, 0.001]
    reg_coeffs = [0.01, 0.001, 0.0001, 0.0]
    wlen = True
    pos = True
    prev_fix = True
    n_pos_units = 50
    outs = ['tanh', 'id']
    n_hidden = 200
    n_layers = 1

    epochs = [20, 40, 60, 80, 100]

    index_selected = int(sys.argv[1]) - 1
    epoch = epochs[index_selected] 

    for model_type in model_types:
        for out_type in outs:
            for lr in lrs:
                for r in rules:
                    for reg_coeff in reg_coeffs:

                        if model_type.startswith('multilayer') and outs=='id':
                            continue

                        if out_type == 'tanh':
                            out = F.tanh
                        elif out_type == 'id':
                            out = F.identity
                        else:
                            raise Exception('Unknown output type: {}'.format(out_type))

                        if model_type == 'linreg':
                            model = LinReg(n_vocab, n_units, loss_func, out, wlen=wlen, pos=pos, prev_fix=prev_fix, n_pos=n_pos, n_pos_units=n_pos_units)
                            train_iter = EyeTrackingSerialIterator(train, batch_size, repeat=True, shuffle=True, wlen=wlen, pos=pos, prev_fix=prev_fix)
                            val_iter = EyeTrackingSerialIterator(val, batch_size, repeat=False, shuffle=True, wlen=wlen, pos=pos, prev_fix=prev_fix)
                        elif model_type == 'context_concat':
                            model = LinRegContextConcat(n_vocab, n_units, loss_func, out, window=window, wlen=wlen, pos=pos, prev_fix=prev_fix, n_pos=n_pos, n_pos_units=n_pos_units)
                            train_iter = EyeTrackingWindowIterator(train, window, batch_size, repeat=True, shuffle=True, wlen=wlen, pos=pos, prev_fix=prev_fix)
                            val_iter = EyeTrackingWindowIterator(val, window, batch_size, repeat=False, shuffle=True, wlen=wlen, pos=pos, prev_fix=prev_fix)
                        elif model_type == 'multilayer':
                            model = Multilayer(n_vocab, n_units, loss_func, out, n_hidden=n_hidden, n_layers=n_layers, wlen=wlen, pos=pos, prev_fix=prev_fix, n_pos=n_pos, n_pos_units=n_pos_units)
                            train_iter = EyeTrackingSerialIterator(train, batch_size, repeat=True, shuffle=True, wlen=wlen, pos=pos, prev_fix=prev_fix)
                            val_iter = EyeTrackingSerialIterator(val, batch_size, repeat=False, shuffle=True, wlen=wlen, pos=pos, prev_fix=prev_fix)
                        elif model_type == 'multilayer_context':
                            model = MultilayerContext(n_vocab, n_units, loss_func, out, n_hidden=n_hidden, n_layers=n_layers, window=1, wlen=wlen, pos=pos, prev_fix=prev_fix, n_pos=n_pos, n_pos_units=n_pos_units)
                            train_iter = EyeTrackingWindowIterator(train, window, batch_size, repeat=True, shuffle=True, wlen=wlen, pos=pos, prev_fix=prev_fix)
                            val_iter = EyeTrackingWindowIterator(val, window, batch_size, repeat=False, shuffle=True, wlen=wlen, pos=pos, prev_fix=prev_fix)
                        else:
                            raise Exception('Unknown model type: {}'.format(model))

                        if gpu >= 0:
                            model.to_gpu()

                        name = '{}_{}_lr{}_reg{}_epochs{}'.format(out_type, rule_name[r], lr, reg_coeff, epoch)
                        print('{}_{}'.format(model_type, name))
                        optimizer = r(lr)
                        optimizer.setup(model)

                        updater = training.StandardUpdater(train_iter, optimizer, converter=convert, device=gpu)
                        trainer = training.Trainer(updater, (epoch, 'epoch'), out=out_path + os.sep + str(epoch) + os.sep + model_type)

                        trainer.extend(extensions.Evaluator(val_iter, model, converter=convert, device=gpu))

                        trainer.extend(extensions.LogReport(log_name='{}.log'.format(name)))
                        trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'validation/main/loss']))

                        #plot_rep = extensions.PlotReport(['main/loss', 'validation/main/loss'], file_name=name + '.pdf')
                        #trainer.extend(plot_rep)

                        trainer.extend(extensions.ProgressBar())
                        trainer.run()

