import collections 
import numpy as np

import sys
sys.path.insert(0, '/media/fede/fedeProSD/eyewordembed/utilities')
import timing
import util

words_path = './dundee_parsed/WORD'
data_path = './dundee_parsed/Tot_fix_dur'
pos_path = './dundee_parsed/UniversalPOS'
wlen_path = './dundee_parsed/UniversalPOS'

def create_vocabulary(words, max_size=None):
	counter = collections.Counter(filter(lambda x: x!='', words))
	counter = counter.most_common(max_size)
	word2id = {k:i for i, (k,v) in enumerate(counter)}
	id2word = {v:k for k,v in word2id.items()}
	return word2id, id2word, counter


if __name__ == "__main__":
	ws = util.load(words_path)
	word2id, id2word, counts = create_vocabulary(ws)

	words, ts = [], []
	for w,t in zip(ws, util.load(data_path)):
		if w=='':
			continue
		words.append(w)
		ts.append(t)
	
	times = np.array(ts).reshape((-1,1))
	times = (times - np.mean(times)) / np.sqrt(np.var(times))

	words_array = np.array([word2id[w] for w in words]).reshape((-1,1))
	dataset = np.hstack((words_array,times))

	np.random.shuffle(dataset)
	train, val, test = np.split(dataset, [int(.8*len(dataset)), int(.9*len(dataset))])

