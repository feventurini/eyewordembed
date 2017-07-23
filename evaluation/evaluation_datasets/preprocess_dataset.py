# constants
simlex_999_src = './SimLex-999/SimLex-999_original.txt'
simlex_999_dest = 'SimLex-999.txt'

ws353_src = './ws353/wordsim353_agreed.txt'
ws353_dest = 'wordsim353.txt'
def preprocess_simlex999(src_file, dest_file):
	with open(src_file,'r') as s:
		with open(dest_file,'w+') as o:
			for line in s:
				tokens = list(map(lambda x: x.lower(), line.split()))
				print('{}\t{}\t{}'.format(tokens[0],tokens[1],tokens[3]), file=o)

def preprocess_ws353(src_file, dest_file):
	with open(src_file,'r') as s:
		with open(dest_file,'w+') as o:
			for line in s:
				if line.startswith('#'):
					continue
				tokens = list(map(lambda x: x.lower(), line.split()))
				print('{}\t{}\t{}'.format(tokens[1],tokens[2],tokens[3]), file=o)

def preprocess_simverb3500(src_file, dest_file):
	with open(src_file,'r') as s:
		with open(dest_file,'w+') as o:
			for line in s:
				if line.startswith('#'):
					continue
				tokens = list(map(lambda x: x.lower(), line.split()))
				print('{}\t{}\t{}'.format(tokens[0], tokens[1],tokens[3]), file=o)

if __name__ == '__main__':
	simlex_999_src = './SimLex-999/SimLex-999_original.txt'
	simlex_999_dest = 'SimLex-999.txt'

	ws353_src = './ws353/wordsim353_agreed.txt'
	ws353_dest = 'wordsim353.txt'

	simverb_3500_src = './SimVerb-3500/SimVerb-3500.txt'
	simverb_3500_dest = 'simverb-3500.txt'

	simverb_3500_src_dev = './SimVerb-3500/SimVerb-500-dev.txt'
	simverb_3500_dest_dev = 'simverb-500-dev.txt'

	simverb_3500_src_test = './SimVerb-3500/SimVerb-3000-test.txt'
	simverb_3500_dest_test = 'simverb-3000-test.txt'

	preprocess_simlex999(simlex_999_src, simlex_999_dest)
	preprocess_ws353(ws353_src, ws353_dest)
	preprocess_simverb3500(simverb_3500_src, simverb_3500_dest)
	preprocess_simverb3500(simverb_3500_src_dev, simverb_3500_dest_dev)
	preprocess_simverb3500(simverb_3500_src_test, simverb_3500_dest_test)