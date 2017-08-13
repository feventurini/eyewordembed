from sklearn.cluster import KMeans
from spherecluster import SphericalKMeans 
from numbers import Number
from pandas import DataFrame
import sys, codecs, numpy
import gensim
from six import iteritems
from nltk.corpus import stopwords

class autovivify_list(dict):
  '''A pickleable version of collections.defaultdict'''
  def __missing__(self, key):
    '''Given a missing key, set initial value to an empty list'''
    value = self[key] = []
    return value

  def __add__(self, x):
    '''Override addition for numeric types when self is empty'''
    if not self and isinstance(x, Number):
      return x
    raise ValueError

  def __sub__(self, x):
    '''Also provide subtraction method'''
    if not self and isinstance(x, Number):
      return -1 * x
    raise ValueError

def build_word_vector_matrix(model_file, n_words):
  '''Return the vectors and labels for the first n_words in vector file'''
  m = gensim.models.Word2Vec.load(model_file)
  numpy_arrays = []
  labels_array = []
  for word, vocab in sorted(iteritems(m.wv.vocab), key=lambda item: item[1].index):
    if word in stopwords.words('english'):
      continue
    row = m.wv.syn0[vocab.index]
    labels_array.append(word)
    numpy_arrays.append(row)

    if n_words != -1 and len(labels_array) == n_words:
      return numpy.array( numpy_arrays ), labels_array

  return numpy.array( numpy_arrays ), labels_array


def find_word_clusters(labels_array, cluster_labels):
  '''Return the set of words in each cluster'''
  cluster_to_words = autovivify_list()
  for c, i in enumerate(cluster_labels):
    cluster_to_words[ i ].append( labels_array[c] )
  return cluster_to_words


if __name__ == "__main__":
  input_vector_file = sys.argv[1] # Vector file input (e.g. glove.6B.300d.txt)
  n_words = int(sys.argv[2]) # Number of words to analyze 
  reduction_factor = float(sys.argv[3]) # Amount of dimension reduction {0,1}
  
  df, labels_array = build_word_vector_matrix(input_vector_file, n_words)

  n_words = len(labels_array)
  n_clusters = int( n_words * reduction_factor ) # Number of clusters to make

  print('Loaded embedding matrix, building {} clusters'.format(n_clusters))
  # kmeans_model = KMeans(init='k-means++', n_clusters=n_clusters, n_init=10)
  kmeans_model = SphericalKMeans(n_clusters=n_clusters, n_init=10)
  kmeans_model.fit(df)

  cluster_labels  = kmeans_model.labels_
  cluster_inertia   = kmeans_model.inertia_
  cluster_to_words  = find_word_clusters(labels_array, cluster_labels)

  for c in cluster_to_words:
    print(cluster_to_words[c])
