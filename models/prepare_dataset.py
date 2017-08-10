import collections 
import numpy as np

import sys
sys.path.insert(0, '../utilities')
import timing
import util
import os
from scipy.ndimage.interpolation import shift
import nltk

np.random.seed(111)

def create_vocabulary(words, max_size=None):
	tokenized_words = filter(lambda x: x!='', words)

	counter = collections.Counter(tokenized_words)
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

def load_dataset(word2id=None, gensim=False, bins=False, surprisal_order=5, tokenize=False, target='tot'):
	assert(target in ['tot','firstfix','firstpass','regress'])

	if bins:
		dundee_folder = '../dataset/dundee_parsed'
		participants_path = os.path.join(dundee_folder,'Participant')
	else:
		dundee_folder = '../dataset/dundee_parsed_gr'

	words_path = os.path.join(dundee_folder,'Word')
	
	if target == 'tot':
		times_path = os.path.join(dundee_folder, 'Tot_fix_dur') #'First_pass_dur')
	elif target == 'firstfix':
		times_path = os.path.join(dundee_folder, 'First_fix_dur') #'First_pass_dur')
	if target == 'firstpass':
		times_path = os.path.join(dundee_folder, 'First_pass_dur') #'First_pass_dur')
	if target == 'regress':
		times_path = os.path.join(dundee_folder, 'Tot_regres_to_dur') #'First_pass_dur')

	ptimes_path = os.path.join(dundee_folder, 'n-1_fix_dur')
	pos_path = os.path.join(dundee_folder,'UniversalPOS')
	freq_path = os.path.join(dundee_folder,'BNC_freq')
	surprisal_path = os.path.join(dundee_folder, '{}gram_surprisal'.format(surprisal_order))
	
	words = util.load(words_path)

	if not word2id:
		word2id, _ = create_vocabulary(words)
		# word2id, _ = create_vocabulary(words)

	lens = [len(w) for w in words]

	pos = util.load(pos_path)
	pos2id, _= create_vocabulary(pos)
	

	targets = util.load(times_path)
	prev_fix_dur = util.load(ptimes_path)
	freqs = util.load(freq_path)
	surprisals = util.load(surprisal_path)

	def extract_vocab(elements):
		return zip(*filter(lambda x: x[0] in word2id, elements))

	def extract_remove_outliers(elements):
		return zip(*filter(lambda x: x[1]<=3.0 and x[1]>=-3.0, elements))

	if bins:
		participants = util.load(participants_path)
		binned_times, n_participants, n_classes = create_bins(targets, participants)
		# prev_binned_times, n_participants, n_classes = create_bins(prev_fix_dur, participants)

		# words, binned_times, lens, pos, prev_binned_times, freqs, surprisals = words[1:], binned_times[1:], lens[1:], pos[1:], binned_times[:-1], freqs[1:], surprisals[1:]
		# ws, ts, ls, ps, pts, fs, ss = zip(*filter(lambda x: x[0] in word2id, zip(words, binned_times, lens, pos, prev_binned_times, freqs, surprisals)))

		ws, ts, ls, ps, pts, fs, ss = extract_vocab(zip(words, binned_times, lens, pos, prev_binned_times, freqs, surprisals))	
		times = np.array(ts).reshape((-1,1))
		ptimes = np.array(pts).reshape((-1,1))
	
	else:
		# words, targets, lens, pos, prev_fix_dur, freqs, surprisals = words[1:], targets[1:], lens[1:], pos[1:], targets[:-1], freqs[1:], surprisals[1:] 
		# ws, ts, ls, ps, pts, fs, ss = zip(*filter(lambda x: x[0] in word2id, zip(words, targets, lens, pos, prev_fix_dur, freqs, surprisals)))
		ws, ts, ls, ps, pts, fs, ss = extract_vocab(zip(words, targets, lens, pos, prev_fix_dur, freqs, surprisals))
		
		ts, mean, std = normalize(ts)
		ws, ts, ls, ps, pts, fs, ss = extract_remove_outliers(zip(ws, ts, ls, ps, pts, fs, ss))
		times = ts
		# times, mean, std = normalize(ts)
		ptimes, _, _ = normalize(pts, mean, std)

	if not gensim:
		words_array = np.array([word2id[w] for w in ws]).reshape((-1,1))
	else:
		words_array = np.array([word2id[w].index for w in ws]).reshape((-1,1))

	freqs = util.load(freq_path)

	pos_array = np.array([pos2id[p] for p in ps]).reshape((-1,1))
	# lens_array, _, _ = normalize(ls)
	# freqs_array, _, _ = normalize(fs)
	# surprisals_array, _, _ = normalize(ss)

	lens_array = np.array(ls).reshape((-1,1))
	freqs_array = np.array(fs).reshape((-1,1))
	surprisals_array = np.array(ss).reshape((-1,1))

	dataset = np.hstack((words_array, times, lens_array, pos_array, ptimes, freqs_array, surprisals_array))
	
	np.random.shuffle(dataset)
	train, val, test = np.split(dataset, [int(.8*len(dataset)), int(.9*len(dataset))])
	# train, val = np.split(dataset, [int(.9*len(dataset))])
	# test = []

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
