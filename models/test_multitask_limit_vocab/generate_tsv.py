import os
import ast
import numpy as np

def extract_log(logname):
	l = {"main/loss": [], "validation/main/loss": []}
	with open(logname,'r') as f:
		try:
			log = ast.literal_eval(f.read())		
		except:
			return None
	for d in log:
		for k in l:
			l[k].append(d[k])
	return l

def listdir(folder, filt=None):
	l = []
	for root, dirs, filenames in os.walk(folder):
		l += map(lambda f: os.path.join(root,f), filenames)
		for d in dirs:
			l += listdir(os.path.join(root, d))
	return list(filter(lambda x: x.endswith(filt), l)) if filt else l

if __name__ == '__main__':
	folder = 'result'

	for subfolder in os.listdir(folder):
		out_name = 'statistics_{}.tsv'.format(subfolder)
		result = []

		for filename in listdir(os.path.join(folder,subfolder), '.log'):	
			l = extract_log(filename)

			filename = ' '.join([filename.split(os.sep)[-2].upper(), filename.split(os.sep)[-1]])
			basename, _ = os.path.splitext(filename)
			if l:
				l['name'] = basename
				if l not in result:
					result.append(l)

		result = sorted(result, key=lambda x: (np.mean(x['validation/main/loss']), x['validation/main/loss'][-1]))
		with open(out_name, 'w+') as out:
			for i in result:
				print(i['name'], file=out)
				del i['name']
				for k in i:
					print('{}\t{}'.format(k,'\t'.join(map(str, i[k]))), file=out)
