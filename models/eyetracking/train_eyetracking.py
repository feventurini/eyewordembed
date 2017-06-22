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

class EyeTrackingWindowIterator(chainer.dataset.Iterator):

    def __init__(self, dataset, window, batch_size, repeat=True, shuffle=True):
        self.words = np.array([b[0] for b in dataset], dtype=np.int32).reshape(-1,1)
        self.times = np.array([b[1] for b in dataset], dtype=np.float32).reshape(-1,1)
        
        self.window = window
        self.batch_size = batch_size
        self._repeat = repeat
        self._shuffle = shuffle

        if self._shuffle:
            self.order = np.random.permutation(len(dataset) - window * 2).astype(np.int32)
        else:
            self.order = np.range(len(dataset) - window * 2).astype(np.int32)

        self.order += window
        self.current_position = 0
        self.epoch = 0
        self.is_new_epoch = False

    def __next__(self):
        if not self._repeat and self.epoch > 0:
            raise StopIteration

        i = self.current_position
        i_end = i + self.batch_size
        position = self.order[i: i_end]
        w = np.random.randint(self.window - 1) + 1
        offset = np.arange(-w, w)
        pos = position[:, None] + offset[None, :]
        context = self.words.take(pos)
        target = self.times.take(position).reshape(-1,1)

        if i_end >= len(self.order):
            if self._shuffle:
                np.random.shuffle(self.order)
            self.epoch += 1
            self.is_new_epoch = True
            self.current_position = 0
        else:
            self.is_new_epoch = False
            self.current_position = i_end

        return context, target

    @property
    def epoch_detail(self):
        return self.epoch + float(self.current_position) / len(self.order)

    def serialize(self, serializer):
        self.current_position = serializer('current_position',
                                           self.current_position)
        self.epoch = serializer('epoch', self.epoch)
        self.is_new_epoch = serializer('is_new_epoch', self.is_new_epoch)
        if self._order is not None:
            serializer('_order', self._order)

class EyeTrackingSerialIterator(chainer.iterators.SerialIterator):

    def __next__(self):
        batch = super(EyeTrackingSerialIterator, self).__next__()
        x = np.array([b[0] for b in batch], dtype=np.int32).reshape(-1,1)
        targets = np.array([b[1] for b in batch], dtype=np.float32).reshape(-1,1)
        return x, targets

    next = __next__


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
        loss = self.loss_func(o, target)
        reporter.report({'loss': loss}, self)
        return loss

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
        loss = self.loss_func(o, target)
        reporter.report({'loss': loss}, self)
        return loss

def convert(batch, device):
    x, targets = batch
    if device >= 0:
        x = cuda.to_gpu(x)
        targets = cuda.to_gpu(targets)
    return x, targets



if __name__ == '__main__':

    word2id_path = '../vocab/word2id'
    id2word_path = '../vocab/id2word'
    word_count_path = '../vocab/word_count_trimmed'
    gigaword_train_folder = '../gigaword_train'
    gigaword_val_folder = '../gigaword_val'

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', default=-1, type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--unit', '-u', default=100, type=int,
                        help='number of units')
    parser.add_argument('--batchsize', '-b', type=int, default=1000,
                        help='learning minibatch size')
    parser.add_argument('--epoch', '-e', default=20, type=int,
                        help='number of epochs to learn')
    parser.add_argument('--model', '-m', choices=['linreg', 'context'],
                    default='linreg',
                    help='model type ("linreg", "context")')
    parser.add_argument('--window', '-w', default=5, type=int,
                    help='window size')
    parser.add_argument('--out-type', '-o', choices=['tanh', 'sigmoid', 'relu', 'id'],
                        default='id',
                        help='activation function type (tanh, sigmoid, relu, identity)')
    parser.add_argument('--reg-coeff', '-r', type=float, default=0.0001,
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

    vocab, index2word, counts, train, val, _ = pd.load_dataset()

    if args.test:
        train = train[:100]
        val = val[:100]

    n_vocab = len(vocab)
    print('n_vocab: %d' % n_vocab)
    print('data length: %d' % len(train))

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
        model = LinReg(n_vocab, n_units, loss_func, out)
        train_iter = EyeTrackingSerialIterator(train, batch_size, repeat=True, shuffle=True)
        val_iter = EyeTrackingSerialIterator(val, batch_size, repeat=False, shuffle=True)
    elif args.model == 'context':
        model = LinRegContext(n_vocab, n_units, loss_func, out)
        train_iter = EyeTrackingWindowIterator(train, window, batch_size, repeat=True, shuffle=True)
        val_iter = EyeTrackingWindowIterator(val, window, batch_size, repeat=False, shuffle=True)
    else:
        raise Exception('Unknown model type: {}'.format(args.model))

    if args.gpu >= 0:
        model.to_gpu()

    optimizer = O.Adam()
    optimizer.setup(model)
    #l2_reg = chainer.optimizer.WeightDecay(args.reg_coeff)
    #optimizer.add_hook(l2_reg, 'l2')

    updater = training.StandardUpdater(train_iter, optimizer, converter=convert, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    trainer.extend(extensions.Evaluator(val_iter, model, converter=convert, device=args.gpu))

    trainer.extend(extensions.LogReport(log_name='log_' + str(args.unit) + '_' + args.out_type))
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'validation/main/loss']))

    trainer.extend(extensions.ProgressBar())
    trainer.run()

    name = 'eyetracking_' + str(args.unit) + '_' + args.out_type
    with open(name + '.model', 'w') as f:
        f.write('%d %d\n' % (len(index2word), args.unit))
        w = cuda.to_cpu(model.embed.W.data)
        for i, wi in enumerate(w):
            v = ' '.join(map(str, wi))
            f.write('%s %s\n' % (index2word[i], v))

    util.save(cuda.to_cpu(model.embed.W.data), name + '_w')