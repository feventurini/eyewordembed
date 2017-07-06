#!/usr/bin/env python
"""Sample script of word embedding model.

This code implements skip-gram model and continuous-bow model.
"""
import argparse
import collections
import numpy as np
import gensim
import logging
import datetime
import os
os.chdir('..')

import sys
sys.path.insert(0, '../utilities')
sys.path.insert(0, '.')

import chainer
from chainer import optimizers as O
from chainer import functions as F
from chainer.training import extension as E
from chainer.training import extensions
import time

import prepare_dataset as pd
from progress_bar import ProgressBarWord2Vec

from eyetracking_batch_iter import EyeTrackingSerialIterator, EyeTrackingWindowIterator
from multitask_batch_iter import BatchIterator
from config import *
from eyetracking import *

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s\r', level=logging.INFO)

class Word2VecExtension(E.Extension):
    trigger = 1, 'iteration'
    default_name = 'word2vec_extension'
    priority = E.PRIORITY_WRITER

    def __init__(self, sentences_iterator, model_eyetracking, model_word2vec, start_alpha=0.025, end_alpha=0.0001):
        self.sentences_iterator = sentences_iterator
        self.model_eyetracking = model_eyetracking
        self.model_word2vec = model_word2vec
        self.start_alpha = start_alpha
        self.next_alpha = start_alpha
        self.end_alpha = end_alpha
        self.n_examples = 0
        self.total_examples = model_word2vec.corpus_count * sentences_iterator.epochs

    def initialize(self, trainer):
        self.model_eyetracking.embed.W.data = self.model_word2vec.wv.syn0

    def __updateLR(self, batch_size):
        self.n_examples += batch_size
        progress = 1.0 * self.n_examples / self.total_examples
        self.alpha = self.next_alpha    
        next_alpha = self.start_alpha - (self.start_alpha - self.end_alpha) * progress
        self.next_alpha = max(self.end_alpha, next_alpha)

    def __call__(self, trainer):

        batch_sentences = self.sentences_iterator.next()
        if batch_sentences == None:
            return

        self.__updateLR(len(batch_sentences))

        # print("BEFORE:")
        # print(self.model_word2vec.wv.syn0)
        # input(self.model_eyetracking.embed.W.data)

        # print(self.alpha, self.next_alpha)        
        self.trained_word_count = self.model_word2vec.train(batch_sentences, epochs=1, total_examples=len(batch_sentences), queue_factor=2, start_alpha=self.alpha, end_alpha=self.next_alpha)

        # print("AFTER:")
        # print(self.model_word2vec.wv.syn0)
        # input(self.model_eyetracking.embed.W.data)

if __name__ == '__main__':
    
    out_folder = 'test_multitask/result'
    rule_name = {O.SGD: 'sgd', O.AdaGrad: 'adagrad', O.Adam: 'adam'}
    i = int(sys.argv[1])

    if i==1:
        config = ('linreg', 'id', 0.001, O.AdaGrad, 0.01)
    elif i==2:
        config = ('linreg', 'id', 0.0, O.AdaGrad, 0.01)
    elif i==3:
        config = ('linreg', 'tanh', 0.001, O.Adam, 0.01)
    elif i==4:
        config = ('context_concat', 'id', 0.001, O.AdaGrad, 0.01)
    elif i==5:
        config = ('context_concat', 'id', 0.01, O.Adam, 0.001)
    elif i==6:
        config = ('multilayer', 'id', 0.0001, O.AdaGrad, 0.01)
    elif i==7:
        config = ('multilayer', 'tanh', 0.0001, O.Adam, 0.1)
    elif i==8:
        config = ('multilayer_context', 'id', 0.001, O.Adam, 0.001)
    elif i==9:
        config = ('multilayer_context', 'id', 0.001, O.AdaGrad, 0.01)

    model_eyetracking_inference, out_type_eyetracking, reg_coeff, learning_rule, lr = config

    name = 'model_{}_{}_{}_{}_{}lr_{}reg_coeff'.format(
        model_word2vec, model_eyetracking_inference, out_type_eyetracking, rule_name[learning_rule], lr, reg_coeff)
    print(name)
    
    if out_type_eyetracking == 'tanh':
        out_eyetracking = F.tanh
    elif out_type_eyetracking == 'relu':
        out_eyetracking = F.relu
    elif out_type_eyetracking == 'sigmoid':
        out_eyetracking = F.sigmoid
    elif out_type_eyetracking == 'id':
        out_eyetracking = F.identity
    else:
        raise Exception('Unknown output type: {}'.format(out_type))

    sentences = gensim.models.word2vec.LineSentence(train_tarball)

    model = gensim.models.word2vec.Word2Vec(sentences=None, size=n_units, alpha=alpha, window=window, min_count=min_count, max_vocab_size=max_vocab_size, 
    sample=sub_sampling, seed=1, workers=n_workers, min_alpha=0.0001, sg=sg, hs=hs, negative=negative, cbow_mean=cbow_mean, 
    iter=epoch, null_word=0, trim_rule=None, sorted_vocab=1, batch_words=10000)

    if not os.path.isdir(out_folder):
        os.makedirs(out_folder)
    if not os.path.isdir(vocab_folder):
        os.makedirs(vocab_folder)

    if os.path.isfile(vocab_folder + os.sep + "init_vocab_" + os.path.basename(train_tarball) + ".model"):
        model.reset_from(gensim.models.Word2Vec.load(vocab_folder + os.sep + "init_vocab_" + os.path.basename(train_tarball) + ".model"))
    else:
        logging.info("Building vocab...")
        model.build_vocab(sentences, keep_raw_vocab=False, trim_rule=None, progress_per=100000, update=False)
        logging.info("Vocabulary built")
        logging.info("Saving initial model with built vocabulary...")
        model.save(vocab_folder + os.sep + "init_vocab_" + os.path.basename(train_tarball) + ".model")

    word2vec_iter = BatchIterator(sentences, epoch, model.corpus_count, batchsize_word2vec)

    vocab, pos2id, train, val, mean, std = pd.load_dataset(model.wv.vocab, gensim=True)

    print('Data samples eyetracking: %d' % len(train))
    print('Data samples word2vec:\t%d' % model.corpus_count)

    b = np.ceil(model.corpus_count/float(batchsize_word2vec))
    batchsize_eyetracking = int(np.floor(len(train)/b))

    print('Batch-size eyetracking: {}'.format(batchsize_eyetracking))
    print('Batch-size word2vec: {}'.format(batchsize_word2vec))
    print('')

    loss_func = F.mean_squared_error

    n_vocab = len(model.wv.vocab)
    n_pos = len(pos2id)
    #print(model.wv.vocab['the'].index)

    if model_eyetracking_inference == 'linreg':
        model_eyetracking = LinReg(n_vocab, n_units, loss_func, out_eyetracking, wlen=lens, pos=pos, n_pos=n_pos, n_pos_units=n_pos_units)
        train_iter = EyeTrackingSerialIterator(train, batchsize_eyetracking, repeat=True, shuffle=True, lens=lens, pos=pos)
        val_iter = EyeTrackingSerialIterator(val, batchsize_eyetracking, repeat=False, shuffle=True, lens=lens, pos=pos)
    elif model_eyetracking_inference == 'context_concat':
        model_eyetracking = LinRegContextConcat(n_vocab, n_units, loss_func, out_eyetracking, window=1, wlen=lens, pos=pos, n_pos=n_pos, n_pos_units=n_pos_units)
        train_iter = EyeTrackingWindowIterator(train, 1, batchsize_eyetracking, repeat=True, shuffle=True, lens=lens, pos=pos)
        val_iter = EyeTrackingWindowIterator(val, 1, batchsize_eyetracking, repeat=False, shuffle=True, lens=lens, pos=pos)
    elif model_eyetracking_inference == 'multilayer':
        model_eyetracking = Multilayer(n_vocab, n_units, loss_func, out_eyetracking, n_hidden=n_hidden, n_layers=n_layers, wlen=lens, pos=pos, n_pos=n_pos, n_pos_units=n_pos_units)
        train_iter = EyeTrackingSerialIterator(train, batchsize_eyetracking, repeat=True, shuffle=True, lens=lens, pos=pos)
        val_iter = EyeTrackingSerialIterator(val, batchsize_eyetracking, repeat=False, shuffle=True, lens=lens, pos=pos)
    elif model_eyetracking_inference == 'multilayer_context':
        model_eyetracking = MultilayerContext(n_vocab, n_units, loss_func, out_eyetracking, n_hidden=n_hidden, n_layers=n_layers, window=1, wlen=lens, pos=pos, n_pos=n_pos, n_pos_units=n_pos_units)
        train_iter = EyeTrackingWindowIterator(train, 1, batchsize_eyetracking, repeat=True, shuffle=True, lens=lens, pos=pos)
        val_iter = EyeTrackingWindowIterator(val, 1, batchsize_eyetracking, repeat=False, shuffle=True, lens=lens, pos=pos)
    else:
        raise Exception('Unknown model type: {}'.format(model))


    if gpu >= 0:
        model.to_gpu()

    optimizer = learning_rule(lr)
    optimizer.setup(model_eyetracking)
    l2_reg = chainer.optimizer.WeightDecay(reg_coeff)
    optimizer.add_hook(l2_reg, 'l2')

    updater = chainer.training.StandardUpdater(train_iter, optimizer, converter=convert, device=gpu)
    trainer = chainer.training.Trainer(updater, (epoch, 'epoch'), out=out_folder)

    trainer.extend(extensions.Evaluator(val_iter, model_eyetracking, converter=convert, device=gpu))

    trainer.extend(extensions.LogReport(log_name=name + '.log'))
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'validation/main/loss']))

    w2v_e = Word2VecExtension(word2vec_iter, model_eyetracking, model)
    trainer.extend(w2v_e)

    trainer.extend(ProgressBarWord2Vec(w2v_e, update_interval=1))

    trainer.run()

    # model.save(out_folder + os.sep + 'multitask_gigaword_' + str(n_units) + 'units_' + model_word2vec + '_' + model_eyetracking_inference  + '_' + 
    #     str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')) + '.model')
    model.save(out_folder + os.sep + 'model_{}_{}_{}_{}_{}lr_{}reg_coeff'.format(
        model_word2vec, model_eyetracking_inference, out_type_eyetracking, rule_name[learning_rule], lr, reg_coeff))