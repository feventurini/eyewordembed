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

import sys
sys.path.insert(0, '../../utilities')
import timing

import chainer
from chainer import optimizers as O
from chainer import functions as F
from chainer.training import extension as E
from chainer.training import extensions
import prepare_dataset as pd
from batch_generator import BatchIterator, EyeTrackingSerialIterator, EyeTrackingWindowIterator
from config import *
from eyetracking import *

class MultiTaskIterator(object):

    def __init__(self, sentences, model_eyetracking, model_word2vec, eyetracking_dataset, interval_multitask, optimizer, multi_task=False):
        self.sentences = sentences
        self.interval_multitask = interval_multitask
        self.updater = training.StandardUpdater(eyetracking_dataset, optimizer, converter=convert, device=gpu)
        self.multi_task = multi_task
        self.model_eyetracking = model_eyetracking
        self.model_word2vec = model_word2vec
        self.counter = 0

    def __update_multitask(self):
        print("BEFORE")
        print(self.model_word2vec.wv.syn0)
        self.model_eyetracking.embed.W.data = self.model_word2vec.wv.syn0
        self.updater.update()
        self.model_word2vec.wv.syn0 = self.model_eyetracking.embed.W.data
        print("AFTER")
        print(self.model_word2vec.wv.syn0)


    def __iter__(self):
        for s in sentences:
            if self.multi_task and self.counter == self.interval_multitask:
                self.counter == 0
                self.__update_multitask()
            self.counter += 1
            yield s

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s\r', level=logging.INFO)
logger = logging.getLogger()

sentences = gensim.models.word2vec.LineSentence(train_tarball)

model = MultiTaskWord2Vec(sentences=None, size=unit, alpha=alpha, window=window, min_count=min_count, max_vocab_size=max_vocab_size, 
sample=sub_sampling, seed=1, workers=n_workers, min_alpha=0.0001, sg=sg, hs=hs, negative=negative, cbow_mean=cbow_mean, 
iter=epoch_word2vec, null_word=0, trim_rule=None, sorted_vocab=1, batch_words=batchsize_word2vec)

if os.path.isfile(out_folder + os.sep + "init_vocab.model"):
	model.reset_from(gensim.models.Word2Vec.load(out_folder + os.sep + "init_vocab.model"))
else:
	logging.info("Building vocab...")
	model.build_vocab(sentences, keep_raw_vocab=False, trim_rule=None, progress_per=100000, update=False)
	logging.info("Vocabulary built")
	logging.info("Saving initial model with built vocabulary...")
	model.save(out_folder + os.sep + "init_vocab.model")

word2vec_iter = BatchIterator(sentences, model.corpus_count, batchsize_word2vec)

train, val = pd.load_dataset(model.wv.vocab)

print('Data samples eyetracking: %d' % len(train))
print('Data samples word2vec:\t%d' % model.corpus_count)

b = model.corpus_count/float(batchsize_word2vec)
batchsize_eyetracking = int(len(train)/b)

print('Batch-size epoch_eyetracking: {}'.format(batchsize_eyetracking))
print('Batch-size word2vec: {}'.format(batchsize_word2vec))

loss_func = F.mean_squared_error

n_vocab = len(model.wv.vocab)
#print(model.wv.vocab['the'].index)

if model_eyetracking == 'linreg':
    model_eyetracking = LinReg(n_vocab, unit, loss_func, out_eyetracking)
    train_iter = EyeTrackingSerialIterator(train, batchsize_eyetracking, repeat=True, shuffle=True)
    val_iter = EyeTrackingSerialIterator(val, batchsize_eyetracking, repeat=False, shuffle=True)
elif model_eyetracking == 'context':
    model_eyetracking = LinRegContext(n_vocab, unit, loss_func, out_eyetracking)
    train_iter = EyeTrackingWindowIterator(train, window_eyetracking, batchsize_eyetracking, repeat=True, shuffle=True)
    val_iter = EyeTrackingWindowIterator(val, window_eyetracking, batchsize_eyetracking, repeat=False, shuffle=True)
else:
    raise Exception('Unknown model type: {}'.format(model))

if gpu >= 0:
    model.to_gpu()

optimizer = O.Adam()
optimizer.setup(model_eyetracking)
l2_reg = chainer.optimizer.WeightDecay(reg_coeff)
optimizer.add_hook(l2_reg, 'l2')

multitask_iter = MultiTaskIterator(sentences, model_eyetracking, model, train_iter, batchsize_word2vec, optimizer, multi_task=True)
model.train(multitask_iter, epochs=model.iter, total_examples=model.corpus_count, queue_factor=2, report_delay=report_delay)
