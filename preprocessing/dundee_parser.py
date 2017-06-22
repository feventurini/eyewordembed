import csv
import string
import sys
sys.path.insert(0, '../utilities')
import timing
import util


def float_or_blank_cast(s):
	try:
		return float(s)
	except:
		return 0.0

def string_lower_nopunct_cast(s):
	return s.strip(string.punctuation).lower()


class DundeeTreebankParser():
	def __init__(self):
		self.map = {}
		self.previous = -1
	
	def initiateParser(self, firstRow):
		c = 0
		print(firstRow)
		for e in firstRow:
			self.map[e.strip()] = c
			c += 1

	def printMapping(self):
		print("\n".join(map(str,self.map.items())))

	def parserow(self, row):
		return [self.wantedParsing[i](row[i].strip()) for i in range(len(row)) if i in self.wantedSet]

	def parseFile(self, csv_path, wantedSet, wantedParsing):
		l = list()
		with open(csv_path,'r') as f:
			r = csv.reader(f, delimiter='\t')
			self.initiateParser(next(r))

			self.wantedSet = set([self.map[w] for w in wantedSet])
			self.wantedParsing = dict([(self.map[k],v) for (k,v) in wantedParsing.items()])

			print(self.wantedSet)
			print(self.wantedParsing)

			print("MAPPING:")
			dtp.printMapping()
			
			result = list()
			for i in range(len(wantedSet)):
				result.append([])

			for row in r:
				if self.previous == row[self.map['WNUM']]:
					continue
				self.previous = row[self.map['WNUM']]
				#print(row)
				temp = dtp.parserow(row)
				for i in range(len(temp)):
					result[i].append(temp[i])

			rev_mapping = {v:k for k,v in self.map.items()}
			return {rev_mapping[j]:result[i] for i,j in enumerate(sorted(self.wantedSet))}
	
	def fileToText(self, file, out):
		l = list()
		with open(csv_path,'r') as f:
			with open(out,'w') as o:
				r = csv.reader(f, delimiter='\t')
				self.initiateParser(next(r))
				for row in r:
					if self.previous == row[self.map['WNUM']]:
						continue
					self.previous = row[self.map['WNUM']]
					l.append(row[self.map["WORD"]])
				print(' '.join(l), file=o)





wantedSet = ['First_pass_dur', 'Mean_fix_dur', 'Tot_fix_dur', 'WLEN', 'WORD', 'WNUM', 'UniversalPOS']
wantedParsing = {'First_pass_dur':float_or_blank_cast, 'Mean_fix_dur':float_or_blank_cast, 'Tot_fix_dur':float_or_blank_cast, 'WLEN':float, 'WORD':string_lower_nopunct_cast, 'WNUM':float, 'UniversalPOS':str}

csv_path = '/media/fede/fedeProSD/eyewordembed/dataset/dundee_eyemovement/treebank/en_Dundee_DLT_freq_goldtok.csv'
dtp = DundeeTreebankParser()
#result = dtp.parseFile(csv_path, wantedSet, wantedParsing)

#for k,v in result.items():
#	util.save(v, '/media/fede/fedeProSD/eyewordembed/models/eyetracking/dundee_parsed/' + k)
dtp.fileToText(csv_path, 'dundee.txt')
