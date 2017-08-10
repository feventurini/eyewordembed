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
    epoch = 10
    window = 1
    out_path = 'test_eyetracking/result_upos_new'

    if gpu >= 0:
        chainer.cuda.get_device_from_id(gpu).use()
        cuda.check_cuda_available()


    batch_size = batchsize
    n_units = unit
    window = window

    rule_name = {O.AdaGrad: 'adagrad'}
    rules = [O.AdaGrad]
    lrs = [0.01]
    reg_coeffs = [0.001]
    n_pos_units = 50
    outs = ['id']
    n_hidden = 200
    n_layerss = [0]
    windows = [0]

    targets = ['tot', 'firstpass', 'firstfix', 'regress']

    epoch = 10
    bins = False

    early_stopping = True

    modes = [(False, False, False, False, False),
                (True, False, False, False, False),
                (True, True, False, False, False),
                (True, True, True, False, False),
                (True, True, True, True, False),
                (True, True, True, True, True)]

    configs = []
    for freq, wlen, prev_fix, pos, surprisal in modes:
        for lr in lrs:
            configs.append((freq, wlen, prev_fix, pos, surprisal, lr))

    print(configs)
    index_selected = int(sys.argv[1]) - 1
    freq, wlen, prev_fix, surprisal, pos, lr = configs[index_selected] 

    for target in targets: 
        for n_layers in n_layerss:
            for window in windows:
                for out_type in outs:
                    for r in rules:
                        for reg_coeff in reg_coeffs:

                            if n_layers > 0 and out_type=='id':
                                continue

                            if bins:
                                vocab, pos2id, n_classes, n_participants, train, val, test = pd.load_dataset(bins=True, target=target)
                            else:
                                vocab, pos2id, train, val, test, mean, std = pd.load_dataset(target=target)
                            
                                print('')
                                print('Mean dataset times: {}'.format(mean))
                                print('Std_dev dataset times: {}'.format(std))

                            n_vocab = len(vocab)
                            n_pos = len(pos2id)
                            print('n_vocab: %d' % n_vocab)
                            print('data length: %d' % len(train))
                            print('n_pos: %d' % n_pos)

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

                            wl = '_wl' if wlen else ''
                            pos = '_pos' if pos else ''
                            freq = '_freq' if freq else ''
                            pf = '_pf' if prev_fix else ''
                            sur = '_sur' if surprisal else ''
                            name = 'eyetracking{}{}{}{}{}'.format(wl, pos, pf, freq, sur)

                            name = '{}_{}layers_{}window_{}_{}_lr{}_reg{}_epochs{}{}{}{}{}{}'.format(target, n_layers, window, out_type, rule_name[r], lr, reg_coeff, epoch, wl, pos, pf, freq, sur)
                            model_type = 'classifier' if bins else 'linreg'
                            print('{}_{}'.format(model_type, name))

                            optimizer = r(lr)
                            optimizer.setup(model)
                            l2_reg = chainer.optimizer.WeightDecay(reg_coeff)
                            optimizer.add_hook(l2_reg, 'l2')

                            updater = training.StandardUpdater(train_iter, optimizer, converter=convert, device=gpu)
                            trainer = training.Trainer(updater, (epoch, 'epoch'), out=out_path + os.sep + str(epoch) + os.sep + model_type + os.sep + target)

                            trainer.extend(extensions.Evaluator(val_iter, model, converter=convert, device=gpu))

                            trainer.extend(extensions.LogReport(log_name='{}.log'.format(name)))

                            if early_stopping:
                                if bins:
                                    trainer.extend(extensions.snapshot_object(model, name + '.eyemodel'), 
                                        trigger=chainer.training.triggers.MaxValueTrigger('validation/main/accuracy', trigger=(1, 'epoch')))
                                else:
                                    trainer.extend(extensions.snapshot_object(model, name + '.eyemodel'), 
                                        trigger=chainer.training.triggers.MinValueTrigger('validation/main/loss', trigger=(1, 'epoch')))
                            
                            if bins:
                                trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'validation/main/loss', 'main/accuracy', 'validation/main/accuracy']))
                            else:
                                trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'validation/main/loss']))

                            trainer.extend(extensions.ProgressBar())
                            trainer.run()

                            if early_stopping:
                                S.load_npz(out_path + os.sep + str(epoch) + os.sep + model_type + os.sep + target + os.sep + name + '.eyemodel', model)
                            if not bins:
                                # p = model.outlayer.W.shape[1]
                                # n = len(val)
                                # def r2_score(x, y, n, p):
                                #     zx = (x-np.mean(x))/np.std(x, ddof=1)
                                #     zy = (y-np.mean(y))/np.std(y, ddof=1)
                                #     r = np.sum(zx*zy)/(len(x)-1)
                                #     return r**2 - (1 - r**2) * (p)/(n-p-1)       

                                def r2_score(x, y):
                                    zx = (x-np.mean(x))/np.std(x, ddof=1)
                                    zy = (y-np.mean(y))/np.std(y, ddof=1)
                                    r = np.sum(zx*zy)/(len(x)-1)
                                    return r**2

                                test_iter = EyetrackingBatchIterator(val, window, batch_size, repeat=False, shuffle=True, wlen=wlen, pos=pos, prev_fix=prev_fix, freq=freq, surprisal=surprisal, bins=bins)
                                test_set = list(test_iter.next())
                                for t in test_iter:
                                    x, y = t
                                    for i in x:
                                        test_set[0][i] = np.concatenate((test_set[0][i],x[i]), axis=0)
                                    test_set[1] = np.concatenate((test_set[1],y), axis=0)

                                test_set = convert(tuple(test_set), gpu)
                                inputs, y = test_set
                                predictions = model.inference(inputs)
                                y = std*y + mean
                                predictions = std*predictions + mean
                                # for t, i in zip(y, predictions):
                                #     print(t, i)
                                r2 = r2_score(y, predictions)
                                # r2 = r2_score(y, predictions, n, p)
                                print('Predicted r_squared coefficient: {}'.format(r2))
                                with open(out_path + os.sep + str(epoch) + os.sep + model_type + os.sep + target + os.sep + name + '.r2', 'w+') as out:
                                    print('Predicted r_squared coefficient: {}'.format(r2), file=out)
