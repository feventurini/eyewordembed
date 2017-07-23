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

from eyetracking_batch_iter import EyetrackingBatchIterator
from multitask_batch_iter import MultitaskBatchIterator
from config_finetune import *
from eyetracking import *

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s\r', level=logging.INFO)
    
if __name__ == '__main__':
    pre_trained_path = '../../GoogleNews-vectors-negative300.bin.gz'
    sentences = gensim.models.word2vec.LineSentence(train_tarball)

    model = gensim.models.word2vec.Word2Vec(sentences=None, size=n_units, alpha=alpha, window=window, min_count=min_count, max_vocab_size=max_vocab_size, 
    sample=sub_sampling, seed=1, workers=n_workers, min_alpha=0.0001, sg=sg, hs=hs, negative=negative, cbow_mean=cbow_mean, 
    iter=5, null_word=0, trim_rule=None, sorted_vocab=1, batch_words=10000)

    if not os.path.isdir(out_folder):
        os.makedirs(out_folder)
    if not os.path.isdir(vocab_folder):
        os.makedirs(vocab_folder)

    if os.path.isfile(vocab_folder + os.sep + "init_vocab_" + os.path.basename(train_tarball) + "_{}.model".format(n_units)):
        model.reset_from(gensim.models.Word2Vec.load(vocab_folder + os.sep + "init_vocab_" + os.path.basename(train_tarball) + ".model"))
    else:
        logging.info("Building vocab...")
        model.build_vocab(sentences, keep_raw_vocab=False, trim_rule=None, progress_per=100000, update=False)
        logging.info("Vocabulary built")
        logging.info("Saving initial model with built vocabulary...")
        model.save(vocab_folder + os.sep + "init_vocab_" + os.path.basename(train_tarball) + "_{}.model".format(n_units))

    if os.path.isfile(os.path.join(out_folder, 'finetuned_gigaword_{}_{}.model'.format(os.path.basename(train_tarball), n_units))):
        model = gensim.models.word2vec.Word2Vec.load(os.path.join(out_folder, 'finetuned_gigaword_{}_{}.model'.format(os.path.basename(train_tarball), n_units)))
    else:
        model.intersect_word2vec_format(pre_trained_path, binary=True)
        model.train(sentences, total_words=None, epochs=model.iter, total_examples=model.corpus_count, queue_factor=2, report_delay=report_delay)
        model.save(os.path.join(out_folder, 'finetuned_gigaword_{}_{}.model'.format(os.path.basename(train_tarball), n_units)))
    
    if bins:
        vocab, pos2id, n_classes, n_participants, train, val = pd.load_dataset(bins=True)
    else:
        vocab, pos2id, train, val, test, mean, std = pd.load_dataset()

    print('Data samples eyetracking: %d' % len(train))

    train_iter = EyetrackingBatchIterator(train, window_eyetracking, batchsize_eyetracking, repeat=True, shuffle=True, wlen=wlen, pos=pos, prev_fix=prev_fix, freq=freq, bins=bins)
    val_iter = EyetrackingBatchIterator(val, window_eyetracking, batchsize_eyetracking, repeat=False, shuffle=True, wlen=wlen, pos=pos, prev_fix=prev_fix, freq=freq, bins=bins)

    if bins:
        loss_func = F.softmax_cross_entropy
    else:
        loss_func = F.mean_squared_error

    n_vocab = len(model.wv.vocab)
    n_pos = len(pos2id)
    #print(model.wv.vocab['the'].index)

    if bins:
        model_eyetracking = EyetrackingClassifier(n_vocab, n_units, n_participants, n_classes, loss_func, out_eyetracking, n_hidden=n_hidden, window=window_eyetracking, n_layers=n_layers, wlen=wlen, pos=pos, prev_fix=prev_fix, freq=freq, n_pos=n_pos, n_pos_units=50)
    else:
        model_eyetracking = EyetrackingLinreg(n_vocab, n_units, loss_func, out_eyetracking, n_hidden=n_hidden, window=window_eyetracking, n_layers=n_layers, wlen=wlen, pos=pos, prev_fix=prev_fix, freq=freq, n_pos=n_pos, n_pos_units=50)

    if gpu >= 0:
        model.to_gpu()

    model_eyetracking.embed.W.data = model.wv.syn0

    optimizer = O.AdaGrad(0.01)
    optimizer.setup(model_eyetracking)
    # l2_reg = chainer.optimizer.WeightDecay(reg_coeff)
    # optimizer.add_hook(l2_reg, 'l2')

    updater = chainer.training.StandardUpdater(train_iter, optimizer, converter=convert, device=gpu)
    trainer = chainer.training.Trainer(updater, (epoch, 'epoch'), out=out_folder)

    trainer.extend(extensions.Evaluator(val_iter, model_eyetracking, converter=convert, device=gpu))

    trainer.extend(extensions.LogReport(log_name='log_' + str(n_units) + '_' + out_type_eyetracking))
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'validation/main/loss']))
    trainer.extend(extensions.ProgressBar())

    trainer.run()

    model.save(os.path.join(out_folder, 'finetuned_gigaword_{}_{}_eyetracking_'.format(os.path.basename(train_tarball), n_units)) + 
        str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')) + '.model')
