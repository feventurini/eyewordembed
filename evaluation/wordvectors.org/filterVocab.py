import sys
import argparse
import os
import gensim

d = {}
for line in open('fullVocab.txt', 'r'):
	d[line.strip()] = 0

parser = argparse.ArgumentParser()
parser.add_argument( 'input', metavar='input', type=str, help="Input word2vec model")
parser.add_argument( '-o', '--output', default='eval_log', help="Log output folder")
parser.add_argument( "-g", "--gensim", required=False, action='store_true', help="Add this option if the model is in gensim format")
parser.add_argument( "-b", "--binary", required=False, action='store_true', help="If word2vec model in binary format, set True, else False")
args = parser.parse_args()

if not os.path.isdir(args.output):
    os.makedirs(args.output)

# load model
if args.gensim:
    model = gensim.models.Word2Vec.load(args.input).wv
else:
    model = gensim.models.KeyedVectors.load_word2vec_format(args.input, binary=args.binary)

model.save_word2vec_format('temp.txt')

with open('temp.txt') as f:
	for line in f.readlines():
		if line.strip().split()[0] in d: print(line.strip())

os.remove('temp.txt')