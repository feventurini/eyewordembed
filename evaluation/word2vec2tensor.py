#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2016 Loreto Parisi <loretoparisi@gmail.com>
# Copyright (C) 2016 Silvio Olivastri <silvio.olivastri@gmail.com>
# Copyright (C) 2016 Radim Rehurek <radim@rare-technologies.com>

"""
USAGE: $ python -m gensim.scripts.word2vec2tensor --input <Word2Vec model file> --output <TSV tensor filename prefix> [--binary] <Word2Vec binary flag>
Where:
    <Word2Vec model file>: Input Word2Vec model
    <TSV tensor filename prefix>: 2D tensor TSV output file name prefix
    <Word2Vec binary flag>: Set True if Word2Vec model is binary. Defaults to False.
Output:
    The script will create two TSV files. A 2d tensor format file, and a Word Embedding metadata file. Both files will
    us the --output file name as prefix
This script is used to convert the word2vec format to Tensorflow 2D tensor and metadata formats for Embedding Visualization
To use the generated TSV 2D tensor and metadata file in the Projector Visualizer, please 
1) Open http://projector.tensorflow.org/. 
2) Choose "Load Data" from the left menu.
3) Select "Choose file" in "Load a TSV file of vectors." and choose you local "_tensor.tsv" file
4) Select "Choose file" in "Load a TSV file of metadata." and choose you local "_metadata.tsv" file

For more information about TensorBoard TSV format please visit:
https://www.tensorflow.org/versions/master/how_tos/embedding_viz/

"""

import os
import sys
import random
import logging
import argparse

import gensim

logger = logging.getLogger(__name__)

def word2vec2tensor(word2vec_model_path, tensor_filename, gensim_model=False, binary=False):
    '''
    Convert Word2Vec mode to 2D tensor TSV file and metadata file
    Args:
        param1 (str): word2vec model file path
        param2 (str): filename prefix
        param2 (bool): set True to use a binary Word2Vec model, defaults to False
    '''
    if gensim_model:
        model = gensim.models.Word2Vec.load(word2vec_model_path).wv
    else:
        model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_model_path, binary=binary)
    
    outfiletsv = tensor_filename + '_tensor.tsv'
    outfiletsvmeta = tensor_filename + '_metadata.tsv'

    outfiletsv_20 = tensor_filename + '_20k_tensor.tsv'
    outfiletsvmeta_20 = tensor_filename + '_20k_metadata.tsv'    
    
    counter = 0
    with open(outfiletsv, 'w+') as file_vector:
        with open(outfiletsvmeta, 'w+') as file_metadata:
            with open(outfiletsv_20, 'w+') as file_vector_20:
                with open(outfiletsvmeta_20, 'w+') as file_metadata_20:
                    for word in model.index2word:
                        vector_row = '\t'.join(map(str, model[word]))
                        file_metadata.write(word + '\n')
                        file_vector.write(vector_row + '\n')
                        if counter < 20000:
                            file_metadata_20.write(word + '\n')
                            file_vector_20.write(vector_row + '\n')
                        counter += 1


    
    logger.info("2D tensor file saved to %s" % outfiletsv)
    logger.info("Tensor metadata file saved to %s" % outfiletsvmeta)

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s', level=logging.INFO)
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s", ' '.join(sys.argv))

    # check and process cmdline input
    program = os.path.basename(sys.argv[0])
    if len(sys.argv) < 2:
        print(globals()['__doc__'] % locals())
        sys.exit(1)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input", required=True,
        help="Input word2vec model")
    parser.add_argument(
        "-o", "--output", required=True,
        help="Output tensor file name prefix")
    parser.add_argument( "-g", "--gensim", 
                        required=False, action='store_true',
                        help="Add this option if the model is in gensim format")
    parser.add_argument( "-b", "--binary", 
                        required=False, action='store_true',
                        help="If word2vec model in binary format, set True, else False")
    args = parser.parse_args()

    word2vec2tensor(args.input, args.output, args.gensim, args.binary)

    logger.info("finished running %s", program)