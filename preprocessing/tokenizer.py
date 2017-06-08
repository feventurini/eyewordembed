import os
import string
import nltk
import nltk.data
import re
from multiprocessing import Pool

punkt = nltk.data.load('tokenizers/punkt/english.pickle')
punc = re.escape(string.punctuation)
reg = re.compile('^[{}]*[{}]$'.format(punc,punc))

src_folder = 'parsed_gigaword'
dest_folder = 'sentence_tokenized_gigaword'

def tokenize_op(filename):
	print(filename)
	tokenizer = SentenceTokenizer(src_folder, filename, dest_folder)
	tokenizer.tokenize_and_save()

class SentenceTokenizer():
	
	def __init__(self, src_folder, filename, dest_folder):
		self.filename = filename
		self.dest_folder = dest_folder
		with open(src_folder + '/' + filename, 'r') as f:
			self.raw = f.read().strip()
	
	def tokenize_and_save(self):
		with open(self.dest_folder + '/' + self.filename, 'w') as f:
			for sentence in punkt.tokenize(self.raw):
				all_tokens = nltk.tokenize.word_tokenize(sentence, preserve_line=True)
				tokens = [t.lower() for t in all_tokens if not reg.match(t)]
				# tokens = [t.lower() for t in all_tokens if not all(char in puncs for char in t)]
				print(' '.join(tokens), file=f)

pool = Pool(4)
pool.map(tokenize_op, os.listdir(src_folder))

