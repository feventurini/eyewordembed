from bs4 import BeautifulSoup
import os
import string
from multiprocessing import Pool

src_folder = 'decompressed_gigaword'
dest_folder = 'parsed_gigaword'

def parse_op(filename):
	print(filename)
	parser = GigawordParser(src_folder, filename, dest_folder)
	parser.parse()

class GigawordParser():
	
	def __init__(self, src_folder, filename, dest_folder):
		self.filename = filename
		with open(src_folder + '/' + filename) as f:
			self.soup = BeautifulSoup(f, "html5lib")
		self.dest_folder = dest_folder
	
	def parse(self):
		with open(self.dest_folder + '/' + self.filename, 'w') as f:
			for p in self.soup.find_all("p"):	
				print(p.string.replace('\n',' ').strip(), file=f)

# .translate(None, string.punctuation).lower(), file=f

pool = Pool(4)
pool.map(parse_op, os.listdir(src_folder))
