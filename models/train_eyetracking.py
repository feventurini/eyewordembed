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
import prepare_dataset as pd
import os

from eyetracking import *
from eyetracking_batch_iter import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', default=-1, type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--unit', '-u', default=100, type=int,
                        help='number of units')
    parser.add_argument('--batchsize', '-b', type=int, default=1000,
                        help='learning minibatch size')
    parser.add_argument('--epoch', '-e', default=20, type=int,
                        help='number of epochs to learn')
    parser.add_argument('--model', '-m', choices=['linreg', 'multilayer', 'context', 'baseline'],
                    default='linreg',
                    help='model type ("linreg", "context")')
    parser.add_argument('--window', '-w', default=1, type=int,
                    help='window size')
    parser.add_argument('--out-type', '-o', choices=['tanh', 'sigmoid', 'relu', 'id'],
                        default='id',
                        help='activation function type (tanh, sigmoid, relu, identity)')
    parser.add_argument('--reg-coeff', '-r', type=float, default=0.001,
                        help='Coefficient for L2 regularization')
    parser.add_argument('--out', default='result',
                        help='Directory to output the result')
    parser.add_argument('--test', dest='test', action='store_true')
    parser.set_defaults(test=False)

    args = parser.parse_args()

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        cuda.check_cuda_available()

    print('GPU: {}'.format(args.gpu))
    print('# unit: {}'.format(args.unit))
    #print('Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('Output type: {}'.format(args.out_type))

    if args.gpu >= 0:
        cuda.get_device_from_id(args.gpu).use()

    vocab, pos2id, train, val, mean, std = pd.load_dataset()
    index2word = {v:k for k,v in vocab.items()}


    print('')
    print('Mean dataset times: {}'.format(mean))
    print('Std_dev dataset times: {}'.format(std))

    # temp = [b[1] for b in train]
    # print('Mean train times: {}'.format(np.mean(temp)))
    # print('Std_dev train times: {}'.format(np.sqrt(np.var(temp))))
    # temp = [b[1] for b in val]
    # print('Mean validation times: {}'.format(np.mean(temp)))
    # print('Std_dev validation times: {}'.format(np.sqrt(np.var(temp))))    

    if args.test:
        train = train[:100]
        val = val[:100]

    n_vocab = len(vocab)
    n_pos = len(pos2id)
    print('n_vocab: %d' % n_vocab)
    print('data length: %d' % len(train))
    print('n_pos: %d' % n_pos)

    loss_func = F.mean_squared_error

    if args.out_type == 'tanh':
        out = F.tanh
    elif args.out_type == 'relu':
        out = F.relu
    elif args.out_type == 'sigmoid':
        out = F.sigmoid
    elif args.out_type == 'id':
        out = F.identity
    else:
        raise Exception('Unknown output type: {}'.format(args.out_type))


    batch_size = args.batchsize
    n_units = args.unit
    window = args.window

    if args.model == 'linreg':
        model = LinReg(n_vocab, n_units, loss_func, out, wlen=True, pos=True, n_pos=n_pos, n_pos_units=50)
        train_iter = EyeTrackingSerialIterator(train, batch_size, repeat=True, shuffle=True, lens=True, pos=True)
        val_iter = EyeTrackingSerialIterator(val, batch_size, repeat=False, shuffle=True, lens=True, pos=True)
    elif args.model == 'multilayer':
        model = Multilayer(n_vocab, n_units, loss_func, out, wlen=True, pos=True, n_layers=1, n_hidden=50, n_pos=n_pos, n_pos_units=50)
        train_iter = EyeTrackingSerialIterator(train, batch_size, repeat=True, shuffle=True, lens=True, pos=True)
        val_iter = EyeTrackingSerialIterator(val, batch_size, repeat=False, shuffle=True, lens=True, pos=True)
    elif args.model == 'context':
        model = LinRegContextConcat(n_vocab, n_units, loss_func, out, wlen=True, pos=True, n_pos=n_pos, n_pos_units=50)
        train_iter = EyeTrackingWindowIterator(train, window, batch_size, repeat=True, shuffle=True, lens=True, pos=True)
        val_iter = EyeTrackingWindowIterator(val, window, batch_size, repeat=False, shuffle=True, lens=True, pos=True)
    elif args.model == 'multilayer_context':
        model = LinRegContextConcat(n_vocab, n_units, loss_func, out, wlen=True, pos=True, n_pos=n_pos, n_pos_units=50)
        train_iter = EyeTrackingWindowIterator(train, window, batch_size, repeat=True, shuffle=True, lens=True, pos=True)
        val_iter = EyeTrackingWindowIterator(val, window, batch_size, repeat=False, shuffle=True, lens=True, pos=True)
    elif args.model == 'baseline':
        model = Baseline(0.0, loss_func)
        train_iter = EyeTrackingSerialIterator(train, batch_size, repeat=True, shuffle=True)
        val_iter = EyeTrackingSerialIterator(val, batch_size, repeat=False, shuffle=True)
    else:
        raise Exception('Unknown model type: {}'.format(args.model))

    if args.gpu >= 0:
        model.to_gpu()

    optimizer = O.Adam()
    optimizer.setup(model)
    l2_reg = chainer.optimizer.WeightDecay(args.reg_coeff)
    optimizer.add_hook(l2_reg, 'l2')

    updater = training.StandardUpdater(train_iter, optimizer, converter=convert, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    trainer.extend(extensions.Evaluator(val_iter, model, converter=convert, device=args.gpu))

    trainer.extend(extensions.LogReport(log_name='log_' + str(args.unit) + '_' + args.out_type))
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'validation/main/loss']))

    trainer.extend(extensions.ProgressBar())
    trainer.run()

    # name = 'eyetracking_' + str(args.unit) + '_' + args.out_type
    # with open(os.path.join(args.out, name + '.model', 'w')) as f:
    #     f.write('%d %d\n' % (len(index2word), args.unit))
    #     w = cuda.to_cpu(model.embed.W.data)
    #     for i, wi in enumerate(w):
    #         v = ' '.join(map(str, wi))
    #         f.write('%s %s\n' % (index2word[i], v))

    # util.save(cuda.to_cpu(model.embed.W.data), os.path.join(args.out, name + '_w'))