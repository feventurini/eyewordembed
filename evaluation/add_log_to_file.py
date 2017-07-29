import ast
import sys

def extract_log(logname, bins=False):
	l = {"main/loss": [], "validation/main/loss": [], 'main/accuracy': [], 'validation/main/accuracy':[] }
	with open(logname,'r') as f:
		try:
			log = ast.literal_eval(f.read())		
		except:
			return None
	for d in log:
		for k in l:
			l[k].append(d[k])
	return l

with open(sys.argv[1], 'a') as out:
	l = extract_log(sys.argv[2])
	for k in l:
		print('{}\t{}'.format(k,'\t'.join(map(str, l[k]))), file=out)
