import sys
import gzip
import numpy
import math
import gensim

from collections import Counter
from operator import itemgetter

''' Read all the word vectors and normalize them '''
def read_word_vectors(filename):    
  word_vecs = {}
  if filename.endswith('.gz'): file_object = gzip.open(filename, 'r')
  else: file_object = open(filename, 'r')

  for line_num, line in enumerate(file_object):
    line = line.strip().lower()
    word = line.split()[0]
    word_vecs[word] = numpy.zeros(len(line.split())-1, dtype=float)
    for index, vec_val in enumerate(line.split()[1:]):
      word_vecs[word][index] = float(vec_val)      
    ''' normalize weight vector '''
    word_vecs[word] /= math.sqrt((word_vecs[word]**2).sum() + 1e-6)        

  sys.stderr.write("Vectors read from: "+filename+" \n")
  return word_vecs

def read_word_vectors_gensim(filename):
  model = gensim.models.Word2Vec.load(filename).wv

  word_vecs = {}
  for i, word in enumerate(model.vocab):
    word_vecs[word] = model[word]
    word_vecs[word] /= math.sqrt((word_vecs[word]**2).sum() + 1e-6)        

  sys.stderr.write("Vectors read from: "+filename+" \n")
  return word_vecs