import os

def listdir(folder, filt=None):
	l = []
	for root, dirs, filenames in os.walk(folder):
		l += map(lambda f: os.path.join(root,f), filenames)
		for d in dirs:
			l += listdir(os.path.join(root, d))
	return list(filter(lambda x: x.endswith(filt), l)) if filt else l

if __name__ == '__main__':
	src_folder = '../models/test_multitask/result_test'
	out_folder = 'eval_log_test'

	for file in listdir(src_folder):
		tar, model, filename = file.split(os.sep)[-3:]

		if filename.endswith('.log') or filename.endswith('.npy') or filename.endswith('.r2'):
			continue
		if not os.path.isfile('{}/{}/{}/{}.log'.format(out_folder, tar, model, filename)):
			bashcommand = 'python evaluate.py {} -o {}/{}/{} -g'.format(file, out_folder, tar, model)
			try:
				os.system(bashcommand)
			except:
				break
		print(tar, filename)
