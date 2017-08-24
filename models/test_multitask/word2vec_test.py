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

os.chdir('..')

import sys
sys.path.insert(0, '../utilities')
sys.path.insert(0, '.')

import timing
from multitask_batch_iter import MultitaskBatchIterator
import datetime

unit = 100
window = 5
epoch = 10
model_types = ['skipgram']
out_type = 'ns'
vocab = 'init_vocab'

tarball_folder = '../dataset/downsampled_gigaword'
n = 10

model_w2v = ['skipgram']
tarballs = ['tokenized_gigaword_{}.tar.bz2'.format(2**(i+1)) for i in range(10,12)]

configurations = []

for tarball in tarballs:
    for model_word2vec in model_types:
        configurations.append((model_word2vec, tarball))

configurations.reverse()
for k, i in enumerate(configurations):
    print(k, i)

i = int(sys.argv[1]) - 1 
model_word2vec, tarball = configurations[i]

out = os.path.join('test_multitask/result_final', tarball.split('.')[0], model_word2vec)
train_tarball = os.path.join(tarball_folder, tarball)

print('# unit: {}'.format(unit))
print('Window: {}'.format(window))
print('# epoch: {}'.format(epoch))
print('Training model: {}'.format(model_word2vec))
print('Output type: {}'.format(out_type))
print('')

if out_type == 'hsm':
    hs = 1
    negative = 0
elif out_type == 'ns':
    hs = 0
    negative = 5
elif out_type == 'original':
    hs = 0
    negative = 0
else:
    raise Exception('Unknown output type: {}'.format(out_type))

if model_word2vec == 'skipgram':
    sg = 1
elif model_word2vec == 'cbow':
    sg = 0
else:
    raise Exception('Unknown model type: {}'.format(model_word2vec))

alpha = 0.025
min_count = 5
max_vocab_size = 400000
sub_sampling = 0.001
n_workers = 3
cbow_mean = 1 # 1:mean, 0:sum
batchsize = 100000

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

sentences = gensim.models.word2vec.LineSentence(train_tarball)

model = gensim.models.word2vec.Word2Vec(sentences=None, size=unit, alpha=alpha, window=window, min_count=min_count, max_vocab_size=max_vocab_size, iter=epoch,
sample=sub_sampling, seed=1, workers=n_workers, min_alpha=0.0001, sg=sg, hs=hs, negative=negative, cbow_mean=cbow_mean, null_word=0, trim_rule=None, sorted_vocab=1)

if not os.path.isdir(out):
    os.makedirs(out)

if os.path.isfile(vocab + os.sep + "init_vocab_" + os.path.basename(train_tarball) + ".model"):
    model.reset_from(gensim.models.Word2Vec.load(vocab + os.sep + "init_vocab_" + os.path.basename(train_tarball) + ".model"))
else:
    logging.info("Building vocab...")
    model.build_vocab(sentences, keep_raw_vocab=False, trim_rule=None, progress_per=100000, update=False)

    # trick to force the words of the dundee corpus in
    save_corpus_count = model.corpus_count
    model.min_count = 0
    dundee = gensim.models.word2vec.LineSentence('../dataset/dundee_vocab.txt')
    model.build_vocab(dundee, keep_raw_vocab=False, trim_rule=None, progress_per=100000, update=True)
    model.corpus_count = save_corpus_count
    #

    logging.info("Vocabulary built")
    logging.info("Saving initial model with built vocabulary...")
    if not os.path.isdir(vocab):
        os.makedirs(vocab)
    model.save(vocab + os.sep + "init_vocab_" + os.path.basename(train_tarball) + ".model")

logging.info("Starting training...")
report_delay = 3.0

word2vec_iter = MultitaskBatchIterator(sentences, epoch, model.corpus_count, batch_size=batchsize, maxsize=5)

start_alpha = 0.025
end_alpha = 0.0001
n_examples = 0 
total_examples = model.corpus_count * epoch

batch_sentences = word2vec_iter.next()

n_examples += len(batch_sentences)
progress = 1.0 * n_examples / total_examples
alpha = start_alpha
next_alpha = start_alpha - (start_alpha - end_alpha) * progress
next_alpha = max(end_alpha, next_alpha)

while batch_sentences:
    model.train(batch_sentences, epochs=1, total_examples=len(batch_sentences), queue_factor=2, start_alpha=alpha, end_alpha=next_alpha)

    batch_sentences = word2vec_iter.next()
    if batch_sentences:
        n_examples += len(batch_sentences)
        progress = 1.0 * n_examples / total_examples
        alpha = next_alpha
        next_alpha = start_alpha - (start_alpha - end_alpha) * progress
        next_alpha = max(end_alpha, next_alpha)

# model.train(sentences, total_words=None, epochs=model.iter, total_examples=model.corpus_count, queue_factor=2, report_delay=report_delay)

model.save(out + os.sep + "std_word2vec.model")
