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
sys.path.insert(0, '../utilities')
import timing

train_tarball = '../gigaword_train.tar.bz2'

parser = argparse.ArgumentParser()
parser.add_argument('--unit', '-u', default=100, type=int,
                    help='number of units')
parser.add_argument('--window', '-w', default=5, type=int,
                    help='window size')
parser.add_argument('--batchsize', '-b', type=int, default=10000,
                    help='learning minibatch size')
parser.add_argument('--epoch', '-e', default=20, type=int,
                    help='number of epochs to learn')
parser.add_argument('--model', '-m', choices=['skipgram', 'cbow'],
                    default='cbow',
                    help='model type ("skipgram", "cbow")')
parser.add_argument('--negative-size', default=5, type=int,
                    help='number of negative samples')
parser.add_argument('--out-type', '-o', choices=['hsm', 'ns', 'original'],
                    default='ns',
                    help='output model type ("hsm": hierarchical softmax, '
                    '"ns": negative sampling, "original": no approximation)')
parser.add_argument('--out', default='result',
                    help='Directory to output the result')
parser.add_argument('--vocab', '-v', default='init_vocab', 
                    help='Directory to save/load the initial vocabulary')
parser.add_argument('--test', dest='test', action='store_true')
parser.set_defaults(test=False)

args = parser.parse_args() 

print('# unit: {}'.format(args.unit))
print('Window: {}'.format(args.window))
print('Minibatch-size: {}'.format(args.batchsize))
print('# epoch: {}'.format(args.epoch))
print('Training model: {}'.format(args.model))
print('Output type: {}'.format(args.out_type))
print('')

if args.out_type == 'hsm':
    hs = 1
    negative = 0
elif args.out_type == 'ns':
    hs = 0
    negative = 5
elif args.out_type == 'original':
    hs = 0
    negative = 0
else:
    raise Exception('Unknown output type: {}'.format(args.out_type))

if args.model == 'skipgram':
    sg = 1
elif args.model == 'cbow':
    sg = 0
else:
    raise Exception('Unknown model type: {}'.format(args.model))

alpha = 0.025
min_count = 5
max_vocab_size = 400000
sub_sampling = 0.001
n_workers = 20
cbow_mean = 1 # 1:mean, 0:sum

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

sentences = gensim.models.word2vec.LineSentence(train_tarball)

model = gensim.models.word2vec.Word2Vec(sentences=None, size=args.unit, alpha=alpha, window=args.window, min_count=min_count, max_vocab_size=max_vocab_size, 
sample=sub_sampling, seed=1, workers=n_workers, min_alpha=0.0001, sg=sg, hs=hs, negative=negative, cbow_mean=cbow_mean, 
iter=args.epoch, null_word=0, trim_rule=None, sorted_vocab=1, batch_words=args.batchsize)

if not os.path.isdir(args.out):
    os.makedirs(args.out)

if os.path.isfile(args.vocab + os.sep + "init_vocab.model"):
    model.reset_from(gensim.models.Word2Vec.load(args.vocab + os.sep + "init_vocab.model"))
else:
    logging.info("Building vocab...")
    model.build_vocab(sentences, keep_raw_vocab=False, trim_rule=None, progress_per=100000, update=False)
    logging.info("Vocabulary built")
    logging.info("Saving initial model with built vocabulary...")
    if not os.path.isdir(args.vocab):
        os.makedirs(args.vocab)
    model.save(args.vocab + os.sep + "init_vocab.model")

logging.info("Starting training...")
report_delay = 3.0
model.train(sentences, total_words=None, epochs=model.iter, total_examples=model.corpus_count, queue_factor=2, report_delay=report_delay)

model.save(args.out + os.sep + "word2vec_gigaword_" + str(args.unit) + "_" + args.model + "_" + args.out_type + ".model")
