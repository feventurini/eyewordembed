import os

def listdir(folder, filt=None):
	l = []
	for root, dirs, filenames in os.walk(folder):
		l += map(lambda f: os.path.join(root,f), filenames)
		for d in dirs:
			l += listdir(os.path.join(root, d))
	return list(filter(lambda x: x.endswith(filt), l)) if filt else l

if __name__ == '__main__':
	src_folder = '../models/test_multitask_limit_vocab/result_final'
	out_folder = 'evaluations/eval_log_word2vec_limit_vocab'

	for file in listdir(src_folder):
		tar, model, filename = file.split(os.sep)[-3:]

		if not filename.endswith('.model'):
			continue
		if not os.path.isfile('{}/{}/{}/{}.log'.format(out_folder, tar, model, filename)):
			bashcommand = 'python evaluate.py {} -o {}/{}/{} -g'.format(file, out_folder, tar, model)
			try:
				os.system(bashcommand)
			except:
				break
		print(tar, filename)
