import gensim
import sys
import collections
from tensorflow.contrib.tensorboard.plugins import projector
import os
import argparse
from smart_open import smart_open
import numpy as np
import tensorflow as tf
from nltk.corpus import stopwords
#logger = logging.getLogger(__name__)

def get_glove_info(glove_file_name):
    """Return the number of vectors and dimensions in a file in GloVe format."""
    with smart_open(glove_file_name) as f:
        num_lines = sum(1 for line in f)
    with smart_open(glove_file_name) as f:
        num_dims = len(f.readline().split()) - 1
    return num_lines, num_dims


def glove2word2vec(glove_input_file, word2vec_output_file):
    """Convert `glove_input_file` in GloVe format into `word2vec_output_file in word2vec format."""
    num_lines, num_dims = get_glove_info(glove_input_file)
    logger.info("converting %i vectors from %s to %s", num_lines, glove_input_file, word2vec_output_file)
    with smart_open(word2vec_output_file, 'wb') as fout:
        fout.write("{0} {1}\n".format(num_lines, num_dims).encode('utf-8'))
        with smart_open(glove_input_file, 'rb') as fin:
            for line in fin:
                fout.write(line)
    return num_lines, num_dims


if __name__ == '__main__': 

    parser = argparse.ArgumentParser()
    parser.add_argument( 'input', metavar='input', type=str, help="Input word2vec model")
    parser.add_argument( "-g", "--gensim", 
                        required=False, action='store_true',
                        help="Add this option if the model is in gensim format")
    parser.add_argument( "-b", "--binary", 
                        required=False, action='store_true',
                        help="If word2vec model in binary format, set True, else False")
    parser.add_argument( "-p", "--port", 
                        required=False, type=int, default=6006,
                        help="Tensorboard port")
    args = parser.parse_args()

    # load model
    if args.gensim:
        model = gensim.models.Word2Vec.load(args.input).wv
    else:
        model = gensim.models.KeyedVectors.load_word2vec_format(args.input, binary=args.binary)

    if not os.path.isdir('log'):
        os.makedirs('log')

    # target_words = [line.strip().lower() for line in open("4000-most-common-english-words-csv.csv")][:2000]
    vocab = collections.Counter()
    with open('../../dataset/dundee_vocab.txt') as f:
        for line in f:
            vocab.update(line.split())
    # target_words = dict(vocab.most_common(2000))
    target_words = vocab
    
    filtered_index2word = list(filter(lambda x: x in target_words, model.index2word))

    embedding = np.empty((len(filtered_index2word), len(model[model.index2word[0]])), dtype=np.float32)
    with open(os.path.join('log', 'metadata.tsv'), 'w+') as file_metadata:
        for i, word in enumerate(filtered_index2word):
            file_metadata.write(word + '\n')
            embedding[i] = model[word]

    """
    DISCLAIMER for plagiarism: the following piece of code was readpted from the first answer to the stackoverflow post:
    https://stackoverflow.com/questions/40849116/how-to-use-tensorboard-embedding-projector
    """
    
    # setup a TensorFlow session
    tf.reset_default_graph()
    sess = tf.InteractiveSession()
    embedding_var = tf.Variable(tf.zeros(embedding.shape), name='embedding')
    place = tf.placeholder(tf.float32, shape=embedding.shape)
    set_x = tf.assign(embedding_var, place, validate_shape=False)
    sess.run(tf.global_variables_initializer())
    sess.run(set_x, feed_dict={place: embedding})

    # create a TensorFlow summary writer
    summary_writer = tf.summary.FileWriter('log')
    config = projector.ProjectorConfig()
    embedding_conf = config.embeddings.add()
    embedding_conf.tensor_name = embedding_var.name
    embedding_conf.metadata_path = os.path.join('log', 'metadata.tsv')
    projector.visualize_embeddings(summary_writer, config)

    # save the model
    saver = tf.train.Saver()
    saver.save(sess, os.path.join('log', "model.ckpt"))

    os.system('tensorboard --logdir=log --port={}'.format(args.port))