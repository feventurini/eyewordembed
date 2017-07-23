import collections 
import numpy as np

import sys
sys.path.insert(0, '../utilities')
import timing
import util
import os
from scipy.ndimage.interpolation import shift

dundee_folder = '../dataset/dundee_parsed_gr'
dundee_folder_bins = '../dataset/dundee_parsed'

np.random.seed(111)

def create_vocabulary(words, max_size=None):
	counter = collections.Counter(filter(lambda x: x!='', words))
	counter = sorted(counter.most_common(max_size))
	word2id = {k:i for i, (k,v) in enumerate(counter)}
	return word2id, counter

def create_bins(times, participants):
	d = dict()
	for t, p in zip(times, participants):
		if p in d:
			d[p].append(t)
		else:
			d[p] = [t]
	
	std_devs = dict()
	avgs = dict()
	for key in d:
		array = np.array(d[key])
		avgs[key] = np.mean(array)
		indices = array == 0
		array[indices] = np.nan
		std_devs[key] = np.nanstd(array)

	def bin(p, t):
		std = std_devs[p]
		avg = avgs[p]
		if t == 0:
			binned = 0
		elif t < avg - 1.0*std:
			binned = 1
		elif t < avg - 0.5*std:
			binned = 2
		elif t < avg + 0.5*std:
			binned = 3
		elif t > avg + 1.0*std:
			binned = 5
		elif t > avg + 0.5*std:
			binned = 4
		return binned

	binned_times = []
	for t, p in zip(times, participants):
		binned_times.append(bin(p,t))

	return binned_times, len(std_devs), 6

def normalize(values, mean=None, std=None):
	res = np.array(values).reshape((-1,1))
	if not mean:
		mean = np.mean(res)
	if not std:
		std = np.sqrt(np.var(res))
	return (res - mean)/std, mean, std

def load_dataset(word2id=None, gensim=False, bins=False):
	if bins:
		dundee_folder = '../dataset/dundee_parsed'
		participants_path = os.path.join(dundee_folder,'Participant')
	else:
		dundee_folder = '../dataset/dundee_parsed_gr'

	words_path = os.path.join(dundee_folder,'WORD')
	times_path = os.path.join(dundee_folder, 'Tot_fix_dur') #'First_pass_dur')
	ptimes_path = os.path.join(dundee_folder, 'n-1_fix_dur')
	pos_path = os.path.join(dundee_folder,'CPOS')
	freq_path = os.path.join(dundee_folder,'BNC_freq')
	
	words = util.load(words_path)
	if not word2id:
		word2id, _ = create_vocabulary(words)

	lens = [len(w) for w in words]

	pos = util.load(pos_path)
	pos2id, _= create_vocabulary(pos)

	tot_fix_dur = util.load(times_path)
	prev_fix_dur = util.load(ptimes_path)
	freqs = util.load(freq_path)
	
	if bins:
		participants = util.load(participants_path)
		binned_times, n_participants, n_classes = create_bins(tot_fix_dur, participants)
		prev_binned_times, n_participants, n_classes = create_bins(prev_fix_dur, participants)

		# words, binned_times, lens, pos, prev_binned_times = words[1:], binned_times[1:], lens[1:], pos[1:], binned_times[:-1]
		# ws, ts, ls, ps, pts, fs = zip(*filter(lambda x: x[0] in word2id, zip(words, tot_fix_dur, lens, pos, prev_fix_dur, freqs)))
		ws, ts, ls, ps, pts, fs = zip(*filter(lambda x: x[0] in word2id, zip(words, binned_times, lens, pos, prev_binned_times, freqs)))
		
		times = np.array(ts).reshape((-1,1))
		ptimes = np.array(pts).reshape((-1,1))
	
	else:
		# words, tot_fix_dur, lens, pos, prev_fix_dur = words[1:], tot_fix_dur[1:], lens[1:], pos[1:], tot_fix_dur[:-1]
		words, tot_fix_dur, lens, pos, prev_fix_dur = words[1:], tot_fix_dur[1:], lens[1:], pos[1:], prev_fix_dur[:-1]
		ws, ts, ls, ps, pts, fs = zip(*filter(lambda x: x[0] in word2id, zip(words, tot_fix_dur, lens, pos, prev_fix_dur, freqs)))
		
		times, mean, std = normalize(ts)
		ptimes, _, _ = normalize(pts, mean, std)

	if not gensim:
		words_array = np.array([word2id[w] for w in ws]).reshape((-1,1))
	else:
		words_array = np.array([word2id[w].index for w in ws]).reshape((-1,1))

	tot_fix_dur = util.load(times_path)
	freqs = util.load(freq_path)

	pos_array = np.array([pos2id[p] for p in ps]).reshape((-1,1))
	lens_array = np.array(ls).reshape((-1,1))
	freqs_array, _, _ = normalize(fs)
	
	dataset = np.hstack((words_array, times, lens_array, pos_array, ptimes, freqs_array))
	
	np.random.shuffle(dataset)
	# train, val, test = np.split(dataset, [int(.8*len(dataset)), int(.9*len(dataset))])
	train, val = np.split(dataset, [int(.9*len(dataset))])
	test = []

	if bins:
		return word2id, pos2id, n_classes, n_participants, train, val, test
	else:
		return word2id, pos2id, train, val, test, mean, std

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
	# create_reference_table()
	_ = load_dataset()
