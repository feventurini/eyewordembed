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
    parser.add_argument('--batchsize', '-batch', type=int, default=1000,
                        help='learning minibatch size')
    parser.add_argument('--epoch', '-e', default=20, type=int,
                        help='number of epochs to learn')
    parser.add_argument('--window', '-w', default=0, type=int,
                    help='window size')
    parser.add_argument('--layers', '-l', default=0, type=int,
                    help='number of layers')
    parser.add_argument('--out-type', '-o', choices=['tanh', 'sigmoid', 'relu', 'id'],
                        default='id',
                        help='activation function type (tanh, sigmoid, relu, identity)')
    parser.add_argument('--reg-coeff', '-r', type=float, default=0.001,
                        help='Coefficient for L2 regularization')
    parser.add_argument( "-prev_fix", "-pf", 
                    required=False, action='store_true',
                    help="Add this option if you want the model to use the previous fixation as input")
    parser.add_argument( "-pos", 
                    required=False, action='store_true',
                    help="Add this option if you want the model to use the pos tag as input")
    parser.add_argument( "-wlen", "-wl", 
                    required=False, action='store_true',
                    help="Add this option if you want the model to use the word length as input")
    parser.add_argument( "-freq", "-f", 
                    required=False, action='store_true',
                    help="Add this option if you want the model to use the frequency as input")
    parser.add_argument( "-surprisal", "-sur",  
                    action='store_true', required=False,
                    help="Add this option if you want the model to use surprisal as a feature")
    parser.add_argument('--surprisal_order', '-order', default=5, type=int,
                    help='order for surprisal lm')
    parser.add_argument( "-bins", "-b", 
                    required=False, action='store_true',
                    help="Whether to use the classifier or linear regression")
    parser.add_argument('--out', default='result',
                        help='Directory to output the result')

    args = parser.parse_args()

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        cuda.check_cuda_available()

    print('GPU: {}'.format(args.gpu))
    print('# unit: {}'.format(args.unit))
    #print('Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('Output type: {}'.format(args.out_type))

    assert(args.surprisal_order in range(2,6))

    if args.gpu >= 0:
        cuda.get_device_from_id(args.gpu).use()

    if args.bins:
        vocab, pos2id, n_classes, n_participants, train, val, test = pd.load_dataset(bins=True, surprisal_order=args.surprisal_order)
    else:
        vocab, pos2id, train, val, test, mean, std = pd.load_dataset(surprisal_order=args.surprisal_order)
    
        print('')
        print('Mean dataset times: {}'.format(mean))
        print('Std_dev dataset times: {}'.format(std))

    index2word = {v:k for k,v in vocab.items()}

    n_vocab = len(vocab)
    n_pos = len(pos2id)
    print('n_vocab: %d' % n_vocab)
    print('data length: %d' % len(train))
    print('n_pos: %d' % n_pos)

    if args.bins:
        loss_func = F.softmax_cross_entropy
    else:
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

    if args.bins:
        model = EyetrackingClassifier(n_vocab, n_units, n_participants, n_classes, loss_func, out, n_hidden=200, window=args.window, n_layers=args.layers, wlen=args.wlen, pos=args.pos, prev_fix=args.prev_fix, freq=args.freq, surprisal=args.surprisal, n_pos=n_pos, n_pos_units=50)
    else:
        model = EyetrackingLinreg(n_vocab, n_units, loss_func, out, n_hidden=200, window=args.window, n_layers=args.layers, wlen=args.wlen, pos=args.pos, prev_fix=args.prev_fix, freq=args.freq, surprisal=args.surprisal, n_pos=n_pos, n_pos_units=10, loss_ratio=1.0)

    train_iter = EyetrackingBatchIterator(train, args.window, batch_size, repeat=True, shuffle=True, wlen=args.wlen, pos=args.pos, prev_fix=args.prev_fix, freq=args.freq, surprisal=args.surprisal, bins=args.bins)
    val_iter = EyetrackingBatchIterator(val, args.window, batch_size, repeat=False, shuffle=True, wlen=args.wlen, pos=args.pos, prev_fix=args.prev_fix, freq=args.freq, surprisal=args.surprisal, bins=args.bins)

    if args.gpu >= 0:
        model.to_gpu()

    optimizer = O.AdaGrad(0.01)
    optimizer.setup(model)
    l2_reg = chainer.optimizer.WeightDecay(args.reg_coeff)
    optimizer.add_hook(l2_reg, 'l2')

    updater = training.StandardUpdater(train_iter, optimizer, converter=convert, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    trainer.extend(extensions.Evaluator(val_iter, model, converter=convert, device=args.gpu))

    trainer.extend(extensions.LogReport(log_name='log_' + str(args.unit) + '_' + args.out_type))

    if args.bins:
        trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'validation/main/loss', 'main/accuracy', 'validation/main/accuracy']))
    else:
        trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'validation/main/loss']))

    trainer.extend(extensions.dump_graph('main/loss', out_name='test_cg.dot'))
    trainer.extend(extensions.ProgressBar())
    trainer.run()


    if not args.bins:
        p = model.outlayer.W.shape[1]
        n = len(np.vstack((train,val)))
        def r2_score(x, y, n, p):
            zx = (x-np.mean(x))/np.std(x, ddof=1)
            zy = (y-np.mean(y))/np.std(y, ddof=1)
            r = np.sum(zx*zy)/(len(x)-1)
            return r**2 - (1 - r**2) * (p)/(n-p-1)       

        test_iter = EyetrackingBatchIterator(np.vstack((train,val)), args.window, batch_size, repeat=False, shuffle=True, wlen=args.wlen, pos=args.pos, prev_fix=args.prev_fix, freq=args.freq, surprisal=args.surprisal, bins=args.bins)
        test_set = list(test_iter.next())
        for t in test_iter:
            x, y = t
            for i in x:
                test_set[0][i] = np.concatenate((test_set[0][i],x[i]), axis=0)
            test_set[1] = np.concatenate((test_set[1],y), axis=0)

        test_set = convert(tuple(test_set), args.gpu)
        inputs, target = test_set
        predictions = model.inference(inputs)
        target = std*target + mean
        predictions = std*predictions + mean
        # for t, i in zip(target, predictions):
        #     print(t, i)
        r2 = r2_score(target, predictions, n, p)
        print('R_squared coefficient: {}'.format(r2))

    wl = '_wl' if args.wlen else ''
    pos = '_pos' if args.pos else ''
    freq = '_freq' if args.freq else ''
    pf = '_pf' if args.prev_fix else ''
    sur = '_sur' if args.surprisal else ''
    name = 'eyetracking{}{}{}{}{}'.format(wl, pos, pf, freq, sur)

    with open(os.path.join(args.out, name + '.w'), 'w') as f:
        f.write('%d %d\n' % (len(index2word), args.unit))
        w = cuda.to_cpu(model.embed.W.data)
        for i, wi in enumerate(w):
            v = ' '.join(map(str, wi))
            f.write('%s %s\n' % (index2word[i], v))

    S.save_npz(os.path.join(args.out, name + '.eyemodel'), model)

