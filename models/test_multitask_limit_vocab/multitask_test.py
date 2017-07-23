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

from eyetracking_batch_iter import EyetrackingBatchIterator
from multitask_batch_iter import MultitaskBatchIterator
from config import *
from eyetracking import *

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s\r', level=logging.WARNING)

class Word2VecExtension(E.Extension):
    trigger = 1, 'iteration'
    default_name = 'word2vec_extension'
    priority = E.PRIORITY_WRITER

    def __init__(self, sentences_iterator, model_eyetracking, model_word2vec, start_alpha=0.025, end_alpha=0.0001, epoch_ratio=1.0):
        self.sentences_iterator = sentences_iterator
        self.model_eyetracking = model_eyetracking
        self.model_word2vec = model_word2vec
        self.epoch_ratio = epoch_ratio

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

        if self.epoch_ratio==1.0 or np.random.random() <= self.epoch_ratio:
            batch_sentences = self.sentences_iterator.next()
            if batch_sentences == None:
                return

            self.__updateLR(len(batch_sentences))
            self.trained_word_count = self.model_word2vec.train(batch_sentences, epochs=1, total_examples=len(batch_sentences), queue_factor=2, start_alpha=self.alpha, end_alpha=self.next_alpha)

if __name__ == '__main__':

    n = 10
    tarball_folder = '../dataset/downsampled_gigaword'
    dundee = '../dataset/dundee.txt'

    model_w2v = ['skipgram', 'cbow']
    tarballs = ['tokenized_gigaword_{}.tar.bz2'.format(2**(i+1)) for i in range(7,n)]
    model_types = ['linreg']
    rule_name = {O.AdaGrad: 'adagrad'}
    rules = [O.AdaGrad]
    lrs = [0.001, 0.01]
    wlen = True
    pos = True
    prev_fix = True
    outs = ['tanh', 'id']
    reg_coeffs = [0.0, 0.001]
    loss_ratios = [1.0, 0.1, 0.01]

    configurations = []

    binss = [True, False]
    model_w2v = ['skipgram']
    tarballs = ['tokenized_gigaword_{}.tar.bz2'.format(2**(i+1)) for i in range(7,n)]
    windows = [2]
    n_layerss = [0]
    rule_name = {O.AdaGrad: 'adagrad'}
    rules = [O.AdaGrad]
    lrs = [0.01]
    wlen = True
    pos = True
    prev_fix = True
    outs = ['tanh', 'id']
    reg_coeffs = [0.0, 0.001]
    ratios = [1.0, 0.1, 0.001]

    configurations = []

    for bins in binss:
        for model_word2vec in model_w2v:
            for tarball in tarballs:
                for window_eyetracking in windows:
                    for n_layers in n_layerss:
                        for out_type in outs:
                            for lr in lrs:
                                for r in rules:
                                    for reg_coeff in reg_coeffs:
                                        for loss_ratio in ratios:
                                            if n_layers > 0 and out_type=='id':
                                                continue
                                            configurations.append((model_word2vec, tarball, bins, window_eyetracking, n_layers, out_type, reg_coeff, r, lr, loss_ratio))


    configurations.reverse()  
    # for k, i in enumerate(configurations):
    #     print(k, i)
    print(len(configurations))

    i = int(sys.argv[1]) - 1
    model_word2vec, tarball, bins, window_eyetracking, n_layers, out_type_eyetracking, reg_coeff, learning_rule, lr, loss_ratio = configurations[i]

    if model_word2vec == 'skipgram':
        sg = 1
    elif model_word2vec == 'cbow':
        sg = 0
    else:
        raise Exception('Unknown model type: {}'.format(model))

    out_folder = os.path.join('test_multitask_limit_vocab/result', tarball.split('.')[0], model_word2vec)
    train_tarball = os.path.join(tarball_folder, tarball)

    k = tarball.split('.')[0].split('_')[-1]

    model_type = 'classifier' if bins else 'linreg'
    name = 'model_{}_{}_{}window_{}layers_{}_{}_{}lr_{}reg_coeff_{}lossratio_{}downsample'.format(
        model_word2vec, model_type, window_eyetracking, n_layers, out_type_eyetracking, rule_name[learning_rule], lr, reg_coeff, loss_ratio, k)
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

    dundee = '../dataset/dundee.txt'
    sentences = gensim.models.word2vec.LineSentence(dundee)

    model = gensim.models.word2vec.Word2Vec(sentences=None, size=n_units, alpha=alpha, window=window, min_count=0, max_vocab_size=max_vocab_size, 
    sample=sub_sampling, seed=1, workers=n_workers, min_alpha=0.0001, sg=sg, hs=hs, negative=negative, cbow_mean=cbow_mean, 
    iter=epoch, null_word=0, trim_rule=None, sorted_vocab=1, batch_words=10000)

    if not os.path.isdir(out_folder):
        os.makedirs(out_folder)
    if not os.path.isdir(vocab_folder):
        os.makedirs(vocab_folder)

    if os.path.isfile(vocab_folder + os.sep + "init_vocab_" + os.path.basename(dundee) + ".model"):
        model.reset_from(gensim.models.Word2Vec.load(vocab_folder + os.sep + "init_vocab_" + os.path.basename(dundee) + ".model"))
    else:
        logging.info("Building vocab...")
        model.build_vocab(sentences, keep_raw_vocab=False, trim_rule=None, progress_per=100000, update=False)
        logging.info("Vocabulary built")
        logging.info("Saving initial model with built vocabulary...")
        model.save(vocab_folder + os.sep + "init_vocab_" + os.path.basename(dundee) + ".model")

    if bins:
        vocab, pos2id, n_classes, n_participants, train, val = pd.load_dataset(model.wv.vocab, gensim=True, bins=True)
    else:
        vocab, pos2id, train, val, test, mean, std = pd.load_dataset(model.wv.vocab, gensim=True)

    model_2 = gensim.models.word2vec.Word2Vec(sentences=None)
    model_2.reset_from(gensim.models.Word2Vec.load(vocab_folder + os.sep + "init_vocab_" + os.path.basename(train_tarball) + ".model"))
    sentences = gensim.models.word2vec.LineSentence(train_tarball)

    print('Data samples eyetracking: %d' % len(train))
    print('Data samples word2vec:\t%d' % model_2.corpus_count)

    b = np.ceil(len(train)/float(batchsize_eyetracking))
    batchsize_word2vec = int(np.floor(model_2.corpus_count/b))
    model.batch_words = np.ceil(batchsize_word2vec/10)

    word2vec_iter = MultitaskBatchIterator(sentences, int(epoch*epoch_ratio), model_2.corpus_count, batchsize_word2vec)
    train_iter = EyetrackingBatchIterator(train, window_eyetracking, batchsize_eyetracking, repeat=True, shuffle=True, wlen=wlen, pos=pos, prev_fix=prev_fix, freq=freq, bins=bins)
    val_iter = EyetrackingBatchIterator(val, window_eyetracking, batchsize_eyetracking, repeat=False, shuffle=True, wlen=wlen, pos=pos, prev_fix=prev_fix, freq=freq, bins=bins)

    print('Batch-size eyetracking: {}'.format(batchsize_eyetracking))
    print('Batch-size word2vec: {}'.format(batchsize_word2vec))
    print('')

    if bins:
        loss_func = F.softmax_cross_entropy
    else:
        loss_func = F.mean_squared_error

    n_vocab = len(model.wv.vocab)
    n_pos = len(pos2id)
    #print(model.wv.vocab['the'].index)

    if bins:
        model_eyetracking = EyetrackingClassifier(n_vocab, n_units, n_participants, n_classes, loss_func, out_eyetracking, n_hidden=n_hidden, window=window_eyetracking, n_layers=n_layers, wlen=wlen, pos=pos, prev_fix=prev_fix, freq=freq, n_pos=n_pos, n_pos_units=50, loss_ratio=loss_ratio)
    else:
        model_eyetracking = EyetrackingLinreg(n_vocab, n_units, loss_func, out_eyetracking, n_hidden=n_hidden, window=window_eyetracking, n_layers=n_layers, wlen=wlen, pos=pos, prev_fix=prev_fix, freq=freq, n_pos=n_pos, n_pos_units=50, loss_ratio=loss_ratio)

    if gpu >= 0:
        model.to_gpu()

    optimizer = learning_rule(lr)
    optimizer.setup(model_eyetracking)
    l2_reg = chainer.optimizer.WeightDecay(reg_coeff)
    optimizer.add_hook(l2_reg, 'l2')

    updater = chainer.training.StandardUpdater(train_iter, optimizer, converter=convert, device=gpu)
    trainer = chainer.training.Trainer(updater, (epoch, 'epoch'), out=out_folder)

    trainer.extend(extensions.Evaluator(val_iter, model_eyetracking, converter=convert, device=gpu))

    trainer.extend(extensions.LogReport(log_name='log_' + str(n_units) + '_' + out_type_eyetracking))

    if bins:
        trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'validation/main/loss', 'main/accuracy', 'validation/main/accuracy']))
    else:
        trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'validation/main/loss']))

    w2v_e = Word2VecExtension(word2vec_iter, model_eyetracking, model, epoch_ratio=epoch_ratio)
    trainer.extend(w2v_e)

    trainer.extend(ProgressBarWord2Vec(w2v_e, update_interval=1))

    trainer.run()

    model.save(out_folder + os.sep + 'limit_vocab_{}.model'.format(name))