import gensim
import sys
from tensorflow.contrib.tensorboard.plugins import projector
import os
import argparse
from smart_open import smart_open
import numpy as np
import tensorflow as tf
#logger = logging.getLogger(__name__)

def word2vec2tensor(word2vec_model_path,tensor_filename, binary=False):
    '''
    Convert Word2Vec mode to 2D tensor TSV file and metadata file
    Args:
        param1 (str): word2vec model file path
        param2 (str): filename prefiembedding_var
        param2 (bool): set True to use a binary Word2Vec model, defaults to False
    '''    
    # model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_model_path, binary=binary)
    model = gensim.models.Word2Vec.load(word2vec_model_path).wv
    outfiletsv = tensor_filename + '_tensor.tsv'
    outfiletsvmeta = tensor_filename + '_metadata.tsv'
    
    with open(outfiletsv, 'w+') as file_vector:
        with open(outfiletsvmeta, 'w+') as file_metadata:
            for word in model.index2word:
                file_metadata.write(word + '\n')
                vector_row = '\t'.join(map(str, model[word]))
                file_vector.write(vector_row + '\n')
    
    logger.info("2D tensor file saved to %s" % outfiletsv)
    logger.info("Tensor metadata file saved to %s" % outfiletsvmeta)


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


# binary = True
# print('Loading model..')
# try:
#     model = gensim.models.KeyedVectors.load_word2vec_format(sys.argv[1], binary=binary)
# except Exception as e:
#     model = gensim.models.Word2Vec.load(sys.argv[1])
# else:
#     model = gensim.models.Word2Vec.load(sys.argv[1])

# print('Model successfully loaded')

# print('Preparing metadata for visualization..')

# # create a list of vectors
# embedding = np.empty((len(model.wv.index2word), model.wv[]), dtype=np.float32)
# for i, word in enumerate(word2vec.words):
#     embedding[i] = word2vec[word]
# summary_writer = tf.summary.FileWriter('log')

# # setup a TensorFlow session
# tf.reset_default_graph()
# sess = tf.InteractiveSession()

# embedding_var = tf.Variable(tf.random_normal(embedding.shape), name='embedding')
# place = tf.placeholder(tf.float32, shape=embedding.shape)
# set_embedding_var = tf.assign(embedding_var, place, validate_shape=False)

# sess.run(tf.global_variables_initializer())
# sess.run(set_embedding_var, feed_dict={place: embedding})

# if not os.path.isdir('log'):
#     os.makedirs('log')

# # write labels
# with open(outfiletsvmeta, 'w+') as file_metadata:
#     for word in model.index2word:
#         file_metadata.write(str(gensim.utils.to_utf8(word) + gensim.utils.to_utf8('\n')))
#         vector_row = '\t'.join(map(str, model[word]))
#         file_vector.write(vector_row + '\n')

# # create a TensorFlow summary writer
# config = projector.ProjectorConfig()
# embedding_conf = config.embeddings.add()
# embedding_conf.tensor_name = embedding_var.name
# embedding_conf.metadata_path = os.path.join('log', 'metadata.tsv')
# projector.visualize_embeddings(summary_writer, config)

# # # save the model
# # saver = tf.train.Saver()
# # saver.save(sess, os.path.join('log', "model.ckpt"))  

# os.system('tensorboard --logdir=log')

if __name__ == '__main__':
    word2vec2tensor(sys.argv[1], 'model', binary=False)    
