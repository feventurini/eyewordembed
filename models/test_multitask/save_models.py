import gensim
folder = ''
import os
import ast
def listdir(folder, filt=None):
	l = []
	for root, dirs, filenames in os.walk(folder):
		l += map(lambda f: os.path.join(root,f), filenames)
		for d in dirs:
			l += listdir(os.path.join(root, d))
	return list(filter(lambda x: x.endswith(filt), l)) if filt else l

if __name__ == '__main__':
	folder = 'result'
	out_folder = '../../evaluation/vectors_multitask_test'
	
	if not os.path.isdir(out_folder):
		os.makedirs(out_folder)
	
	for filename in listdir(folder):
		if filename.endswith('.log') or filename.endswith('.npy'):
			continue
		dataset = filename.split(os.sep)[1]
		name = '{}_{}'.format(dataset, filename.split(os.sep)[-1])
		if not os.path.isfile(os.path.join(out_folder,'{}.txt'.format(name))):
			m = gensim.models.word2vec.Word2Vec.load(filename)
			m.wv.save_word2vec_format(os.path.join(out_folder,'{}.txt'.format(name)))
		print(name)
		
