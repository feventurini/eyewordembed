import collections 
import numpy as np

import sys
sys.path.insert(0, '../utilities')
import timing
import util
import os

dundee_folder = '../dataset/dundee_parsed_gr'
words_path = os.path.join(dundee_folder,'WORD')
data_path = os.path.join(dundee_folder,'Tot_fix_dur')
pos_path = os.path.join(dundee_folder,'CPOS')

np.random.seed(111)

def create_vocabulary(words, max_size=None):
	counter = collections.Counter(filter(lambda x: x!='', words))
	counter = sorted(counter.most_common(max_size))
	word2id = {k:i for i, (k,v) in enumerate(counter)}
	return word2id, counter

def load_dataset(word2id=None, gensim=False):
	words = util.load(words_path)
	if not word2id:
		word2id, _ = create_vocabulary(words)

	lens = [len(w) for w in words]

	pos = util.load(pos_path)
	pos2id, _= create_vocabulary(pos)

	tot_fix_dur = util.load(data_path)
	ws, ts, ls, ps = zip(*filter(lambda x: x[0] in word2id, zip(words, tot_fix_dur, lens, pos)))
	
	times = np.array(ts).reshape((-1,1))
	mean = np.mean(times)
	std = np.sqrt(np.var(times))

	times = (times - mean) / std

	if not gensim:
		words_array = np.array([word2id[w] for w in ws]).reshape((-1,1))
	else:
		words_array = np.array([word2id[w].index for w in ws]).reshape((-1,1))

	pos_array = np.array([pos2id[p] for p in ps]).reshape((-1,1))
	lens_array = np.array(ls).reshape((-1,1))

	dataset = np.hstack((words_array,times,lens_array,pos_array))

	np.random.shuffle(dataset)
	train, val = np.split(dataset, [int(.9*len(dataset))])
	
	return word2id, pos2id, train, val, mean, std

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
