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
import time
from progress_bar import ProgressBarWord2Vec

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s\r', level=logging.CRITICAL)

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

        # print("BEFORE:")
        # print(self.model_word2vec.wv.syn0)
        # input(self.model_eyetracking.embed.W.data)

        self.trained_word_count = self.model_word2vec.train(batch_sentences, epochs=1, total_examples=len(batch_sentences), queue_factor=2)

        # print("AFTER:")
        # print(self.model_word2vec.wv.syn0)
        # input(self.model_eyetracking.embed.W.data)


sentences = gensim.models.word2vec.LineSentence(train_tarball)

model = gensim.models.word2vec.Word2Vec(sentences=None, size=unit, alpha=alpha, window=window, min_count=min_count, max_vocab_size=max_vocab_size, 
sample=sub_sampling, seed=1, workers=n_workers, min_alpha=0.0001, sg=sg, hs=hs, negative=negative, cbow_mean=cbow_mean, 
iter=epoch, null_word=0, trim_rule=None, sorted_vocab=1, batch_words=10000)

if os.path.isfile(out_folder + os.sep + "init_vocab.model"):
	model.reset_from(gensim.models.Word2Vec.load(out_folder + os.sep + "init_vocab.model"))
else:
	logging.info("Building vocab...")
	model.build_vocab(sentences, keep_raw_vocab=False, trim_rule=None, progress_per=100000, update=False)
	logging.info("Vocabulary built")
	logging.info("Saving initial model with built vocabulary...")
	model.save(out_folder + os.sep + "init_vocab.model")

word2vec_iter = BatchIterator(sentences, epoch, model.corpus_count, batchsize_word2vec)

train, val = pd.load_dataset(model.wv.vocab)

print('Data samples eyetracking: %d' % len(train))
print('Data samples word2vec:\t%d' % model.corpus_count)

b = np.ceil(model.corpus_count/float(batchsize_word2vec))
batchsize_eyetracking = int(np.floor(len(train)/b))

print('Batch-size eyetracking: {}'.format(batchsize_eyetracking))
print('Batch-size word2vec: {}'.format(batchsize_word2vec))
print('')

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

updater = training.StandardUpdater(train_iter, optimizer, converter=convert, device=gpu)
trainer = training.Trainer(updater, (epoch, 'epoch'), out=out_folder)

trainer.extend(extensions.Evaluator(val_iter, model_eyetracking, converter=convert, device=gpu))

trainer.extend(extensions.LogReport(log_name='log_' + str(unit) + '_' + out_type_eyetracking))
trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'validation/main/loss']))

w2v_e = Word2VecExtension(word2vec_iter, model_eyetracking, model)
trainer.extend(w2v_e)

trainer.extend(ProgressBarWord2Vec(w2v_e, update_interval=1))

trainer.run()
