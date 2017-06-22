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
import prepare_dataset as pd
from batch_generator import BatchIterator, EyeTrackingSerialIterator, EyeTrackingWindowIterator
from config import *
from eyetracking import *

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

sentences = gensim.models.word2vec.LineSentence(train_tarball)

model = gensim.models.word2vec.Word2Vec(sentences=None, size=unit, alpha=alpha, window=window, min_count=min_count, max_vocab_size=max_vocab_size, 
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

logging.info('data samples eyetracking: %d' % len(train))
logging.info('data samples word2vec:\t%d' % model.corpus_count)

b = model.corpus_count/float(batchsize_word2vec)
batchsize_eyetracking = int(len(train)/b)

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

train_count = len(train)
val_count = len(val)

sum_accuracy = 0
sum_loss = 0
counter = 0
model_eyetracking.embed.W.data = model.wv.syn0

while train_iter.epoch < epoch_eyetracking and word2vec_iter.epoch < epoch_word2vec:

    # print('BEFORE EYETRACKING')
    # print(model_eyetracking.embed.W.data)
    # input()

    batch = train_iter.next()
    counter += 1
    x_array, t_array = convert(batch, gpu)
    x = chainer.Variable(x_array)
    t = chainer.Variable(t_array)
    optimizer.update(model_eyetracking, x, t)
    sum_loss += float(model_eyetracking.loss.data) * len(t.data)

    # print('AFTER EYETRACKING')
    # print(model_eyetracking.embed.W.data)
    # input()
    #if train_iter.is_new_epoch:
    if counter % 10 == 0:
        print('epoch: ', train_iter.epoch)
        print('train mean loss: {}'.format(sum_loss / train_count))
        # evaluation
        sum_loss = 0
        for batch in val_iter:
            x_array, t_array = convert(batch, gpu)
            x = chainer.Variable(x_array)
            t = chainer.Variable(t_array)
            loss = model_eyetracking(x, t)
            sum_loss += float(loss.data) * len(t.data)
   
        val_iter.reset()
        print('val mean  loss: {}'.format(
            sum_loss / val_count))
        sum_loss = 0

    #model.wv.syn0 = model_eyetracking.embed.W.data

    # print('BEFORE Word2Vec')
    # print(model.wv.syn0)
    # input()
    batch_sentences = word2vec_iter.next()
    model.train(batch_sentences, total_words=None, epochs=1, total_examples=len(batch_sentences), queue_factor=2, report_delay=report_delay)
    # print('AFTER Word2Vec')
    # print(model.wv.syn0)
    # input()
    

