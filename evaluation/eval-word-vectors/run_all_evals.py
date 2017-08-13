import os

def listdir(folder, filt=None):
	l = []
	for root, dirs, filenames in os.walk(folder):
		l += map(lambda f: os.path.join(root,f), filenames)
		for d in dirs:
			l += listdir(os.path.join(root, d))
	return set(filter(lambda x: x.endswith(filt), l)) if filt else l

for file in listdir('../../models/test_multitask', '.model'):
	vocab = 'full_vocab'
	tokens = file.split(os.sep)
	if tokens[-1].startswith('std'):
		downsample = tokens[5]
		model = tokens[6].split('_')[-1]
		if model!='skipgram':
			continue
		out_name = 'evals/{}/{}_{}_{}.txt'.format(vocab, 'std', model, downsample)
	else:
		typ = '_'.join(tokens[3].split('_')[1:])
		downsample = tokens[5]
		model = tokens[6].split('_')[-1]
		if model!='skipgram':
			continue
		reg = 'reg' if tokens[-1].split('_')[-4] == '0.001reg' else 'noreg'
		if reg!='reg':
			continue
		out_name = 'evals/{}/{}_{}_{}_{}.txt'.format(vocab, typ, model, downsample, reg)
		
	if not os.path.exists('evals/{}'.format(vocab)):
		os.makedirs('evals/{}'.format(vocab))

	if not os.path.exists(out_name):
		print(out_name)
		os.system('python2 all_wordsim.py {} data/word-sim > ./{}'.format(file, out_name))