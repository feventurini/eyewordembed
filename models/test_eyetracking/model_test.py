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


    batch_size = batchsize
    n_units = unit
    window = window

    rule_name = {O.AdaGrad: 'adagrad', O.Adam: 'adam'}
    rules = [O.AdaGrad, O.Adam]
    lrs = [0.01, 0.001, 0.0001]
    reg_coeffs = [0.001, 0.01, 0.0]
    wlen = True
    pos = True
    prev_fix = True
    freq = True
    surprisal = True
    n_pos_units = 50
    outs = ['tanh', 'id']
    n_hidden = 200
    n_layerss = [0, 1, 2]
    windows = [0, 1, 2]

    epochs_model = [(20, True), (20, False), (60, True), (60, False), (100, True), (100, False)]

    configs = []
    for e, m in epochs_model:
        for lr in lrs:
            configs.append((e,m,lr))

    print(configs)
    index_selected = int(sys.argv[1]) - 1
    epoch, bins, lr = configs[index_selected] 

    if bins:
        vocab, pos2id, n_classes, n_participants, train, val, test = pd.load_dataset(bins=True)
    else:
        vocab, pos2id, train, val, test, mean, std = pd.load_dataset()
    
        print('')
        print('Mean dataset times: {}'.format(mean))
        print('Std_dev dataset times: {}'.format(std))

    n_vocab = len(vocab)
    n_pos = len(pos2id)
    print('n_vocab: %d' % n_vocab)
    print('data length: %d' % len(train))
    print('n_pos: %d' % n_pos)

    for n_layers in n_layerss:
        for window in windows:
            for out_type in outs:
                for r in rules:
                    for reg_coeff in reg_coeffs:

                        if n_layers > 0 and out_type=='id':
                            continue

                        if out_type == 'tanh':
                            out = F.tanh
                        elif out_type == 'id':
                            out = F.identity
                        else:
                            raise Exception('Unknown output type: {}'.format(out_type))

                        if bins:
                            loss_func = F.softmax_cross_entropy
                        else:
                            loss_func = F.mean_squared_error

                        if bins:
                            model = EyetrackingClassifier(n_vocab, n_units, n_participants, n_classes, loss_func, out, n_hidden=n_hidden, window=window, n_layers=n_layers, wlen=wlen, pos=pos, prev_fix=prev_fix, freq=freq, surprisal=surprisal, n_pos=n_pos, n_pos_units=50)
                        else:
                            model = EyetrackingLinreg(n_vocab, n_units, loss_func, out, n_hidden=n_hidden, window=window, n_layers=n_layers, wlen=wlen, pos=pos, prev_fix=prev_fix, freq=freq, surprisal=surprisal, n_pos=n_pos, n_pos_units=50)

                        train_iter = EyetrackingBatchIterator(train, window, batch_size, repeat=True, shuffle=True, wlen=wlen, pos=pos, prev_fix=prev_fix, freq=freq, surprisal=surprisal, bins=bins)
                        val_iter = EyetrackingBatchIterator(val, window, batch_size, repeat=False, shuffle=True, wlen=wlen, pos=pos, prev_fix=prev_fix, freq=freq, surprisal=surprisal, bins=bins)

                        if gpu >= 0:
                            model.to_gpu()

                        name = '{}layers_{}window_{}_{}_lr{}_reg{}_epochs{}'.format(n_layers, window, out_type, rule_name[r], lr, reg_coeff, epoch)
                        model_type = 'classifier' if bins else 'linreg'
                        print('{}_{}'.format(model_type, name))

                        optimizer = r(lr)
                        optimizer.setup(model)
                        l2_reg = chainer.optimizer.WeightDecay(reg_coeff)
                        optimizer.add_hook(l2_reg, 'l2')

                        updater = training.StandardUpdater(train_iter, optimizer, converter=convert, device=gpu)
                        trainer = training.Trainer(updater, (epoch, 'epoch'), out=out_path + os.sep + str(epoch) + os.sep + model_type)

                        trainer.extend(extensions.Evaluator(val_iter, model, converter=convert, device=gpu))

                        trainer.extend(extensions.LogReport(log_name='{}.log'.format(name)))
                        
                        if bins:
                            trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'validation/main/loss', 'main/accuracy', 'validation/main/accuracy']))
                        else:
                            trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'validation/main/loss']))

                        trainer.extend(extensions.ProgressBar())
                        trainer.run()

