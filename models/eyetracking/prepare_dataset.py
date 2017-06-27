import collections 
import numpy as np

import sys
sys.path.insert(0, '/media/fede/fedeProSD/eyewordembed/utilities')
import timing
import util
import os

dundee_folder = '/media/fede/fedeProSD/eyewordembed/dataset/dundee_parsed_gr'
words_path = os.path.join(dundee_folder,'WORD')
data_path = os.path.join(dundee_folder,'Tot_fix_dur')
pos_path = os.path.join(dundee_folder,'UniversalPOS')

def create_vocabulary(words, max_size=None):
	counter = collections.Counter(filter(lambda x: x!='', words))
	counter = counter.most_common(max_size)
	word2id = {k:i for i, (k,v) in enumerate(counter)}
	id2word = {v:k for k,v in word2id.items()}
	return word2id, id2word, counter

def load_dataset():
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

	return word2id, id2word, counts, train, val, test

def create_reference_table():
	words = util.load(words_path)
	times = np.array(util.load(data_path))
	pos = util.load(pos_path)

	table = collections.defaultdict(dict)
	count = collections.defaultdict(dict)
	for w,t,p in zip(words,times,pos):
		if w=="":
			continue
		l = len(w)
		if p not in table or l not in table[p]:
			table[p][l] = t
			count[p][l] = 1
		else:
			table[p][l] += t
			count[p][l] += 1

	for p in table:
		for l in table[p]:
			table[p][l] = table[p][l] / float(count[p][l])

	print(table)

	util.save(table, 'pos_len_table')


if __name__ == '__main__':
	create_reference_table()
	_ = load_dataset()
