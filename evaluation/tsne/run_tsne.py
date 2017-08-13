from tsne import tsne
import gensim
import pylab as Plot
import sys
import os
import numpy as Math

model = gensim.models.Word2Vec.load(sys.argv[1])

vocab = set()
with open('../../dataset/dundee_vocab.txt') as f:
    for line in f:
        vocab.update(line.split())

target_words = [line.strip().lower() for line in open("4000-most-common-english-words-csv.csv")][:2000]
target_words = list(filter(lambda x: x in vocab, target_words))
rows = [model.wv.vocab[word].index for word in target_words if word in model.wv.vocab]
target_matrix = model.wv.syn0[rows,:]

reduced_matrix = tsne(target_matrix, 2)

Plot.figure(figsize=(200, 200), dpi=100)
max_x = Math.amax(reduced_matrix, axis=0)[0]
max_y = Math.amax(reduced_matrix, axis=0)[1]
Plot.xlim((-max_x,max_x))
Plot.ylim((-max_y,max_y))

Plot.scatter(reduced_matrix[:, 0], reduced_matrix[:, 1], 20);

for row_id in range(0, len(rows)):
    target_word = model.wv.index2word[rows[row_id]]
    x = reduced_matrix[row_id, 0]
    y = reduced_matrix[row_id, 1]
    Plot.annotate(target_word, (x,y))

Plot.savefig(os.path.basename(sys.argv[1]) + ".png")