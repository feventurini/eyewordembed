import sys
sys.path.insert(0, '../preprocessing')

import parallel_vocab_creator as pvc
import numpy as np

word_count = pvc.load('word_count')
word2id = pvc.load('word2id')
id2word = pvc.load('id2word')

print(id2word['<UNK>'])

