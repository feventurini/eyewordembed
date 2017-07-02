#!/usr/bin/env python
"""Sample script of word embedding model.

This code implements skip-gram model and continuous-bow model.
"""
import argparse
import collections
import numpy as np
import gensim
import logging
import os
import datetime
import sys
sys.path.insert(0, '../utilities')
import timing

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

    def __init__(self, sentences_iterator, model_eyetracking, model_word2vec):
        self.sentences_iterator = sentences_iterator
        self.model_eyetracking = model_eyetracking
        self.model_word2vec = model_word2vec
        self.alpha = alpha
        self.progress = batchsize_word2vec / model_word2vec.corpus_count
        self.min_alpha = self.alpha - self.progress*self.alpha
        self.speed = 0
        
    def initialize(self, trainer):
        self.model_eyetracking.embed.W.data = self.model_word2vec.wv.syn0

    def __updateLR(self):
        self.alpha = self.min_alpha 
        self.min_alpha = self.alpha - self.progress*self.alpha
      
    def __call__(self, trainer):
        self.model_word2vec.alpha = self.alpha
        self.model_word2vec.min_alpha = self.min_alpha
        self.__updateLR()

        batch_sentences = self.sentences_iterator.next()
        if batch_sentences == None:
            return

        print("BEFORE:")
        print(self.model_word2vec.wv.syn0)
        input(self.model_eyetracking.embed.W.data)

        self.trained_word_count = self.model_word2vec.train(batch_sentences, epochs=1, total_examples=len(batch_sentences), queue_factor=2)

        print("AFTER:")
        print(self.model_word2vec.wv.syn0)
        input(self.model_eyetracking.embed.W.data)

if __name__ == '__main__':
    
    sentences = gensim.models.word2vec.LineSentence(train_tarball)

    model = gensim.models.word2vec.Word2Vec(sentences=None, size=n_units, alpha=alpha, window=window, min_count=min_count, max_vocab_size=max_vocab_size, 
    sample=sub_sampling, seed=1, workers=n_workers, min_alpha=0.0001, sg=sg, hs=hs, negative=negative, cbow_mean=cbow_mean, 
    iter=epoch, null_word=0, trim_rule=None, sorted_vocab=1, batch_words=10000)

    if not os.path.isdir(out_folder):
        os.makedirs(out_folder)
    if not os.path.isdir(vocab_folder):
        os.makedirs(vocab_folder)

    if os.path.isfile(vocab_folder + os.sep + "init_vocab.model"):
        model.reset_from(gensim.models.Word2Vec.load(vocab_folder + os.sep + "init_vocab.model"))
    else:
        logging.info("Building vocab...")
        model.build_vocab(sentences, keep_raw_vocab=False, trim_rule=None, progress_per=100000, update=False)
        logging.info("Vocabulary built")
        logging.info("Saving initial model with built vocabulary...")
        model.save(vocab_folder + os.sep + "init_vocab.model")

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
    elif model_eyetracking_inference == 'context':
        model_eyetracking = LinRegContextConcat(n_vocab, n_units, loss_func, out_eyetracking, wlen=True, pos=True, n_pos=n_pos, n_pos_units=50)
        train_iter = EyeTrackingWindowIterator(train, window, batchsize_eyetracking, repeat=True, shuffle=True, lens=lens, pos=pos)
        val_iter = EyeTrackingWindowIterator(val, window, batchsize_eyetracking, repeat=False, shuffle=True, lens=lens, pos=pos)
    else:
        raise Exception('Unknown model type: {}'.format(model))

    if gpu >= 0:
        model.to_gpu()

    optimizer = O.Adam()
    optimizer.setup(model_eyetracking)
    l2_reg = chainer.optimizer.WeightDecay(reg_coeff)
    optimizer.add_hook(l2_reg, 'l2')

    updater = chainer.training.StandardUpdater(train_iter, optimizer, converter=convert, device=gpu)
    trainer = chainer.training.Trainer(updater, (epoch, 'epoch'), out=out_folder)

    trainer.extend(extensions.Evaluator(val_iter, model_eyetracking, converter=convert, device=gpu))

    trainer.extend(extensions.LogReport(log_name='log_' + str(n_units) + '_' + out_type_eyetracking))
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'validation/main/loss']))

    w2v_e = Word2VecExtension(word2vec_iter, model_eyetracking, model)
    trainer.extend(w2v_e)

    trainer.extend(ProgressBarWord2Vec(w2v_e, update_interval=1))

    trainer.run()

    model.save(out_folder + os.sep + 'multitask_gigaword_' + str(n_units) + 'units_' + model_word2vec + '_' + model_eyetracking_inference  + '_' + 
        str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')) + '.model')