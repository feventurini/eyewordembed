import csv
import string
import sys
sys.path.insert(0, '../utilities')
import timing
import util
import os


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
			result.append([])

			def add_to_res(row, previous_fixation):
				temp = dtp.parserow(row)
				for i in range(len(temp)):
					result[i].append(temp[i])
				i += 1
				result[i].append(previous_fixation)

			index_tot_fix_dur = [i for i,j in enumerate(sorted(self.wantedSet)) if j==self.map['Tot_fix_dur']][0]
			count_zero = 0

			first = True
			previous_fixation = -1
			for row in r:
				try:
					tot_fix_dur = float(row[self.map['Tot_fix_dur']])
				except:
					tot_fix_dur = 0.0

				# Remove cases of trackloss, i.e., sequences of four adjacent words
				# that are not fixated.
				if tot_fix_dur != 0:
					if count_zero >=4:
						for i in range(1, count_zero + 1):
							if result[index_tot_fix_dur][-i] == 0:
								for l in result:
									del l[-i]
					count_zero = 0
				else:
					count_zero += 1

				# Remove cases where the reading time in Dundee has been calculated
				# erroneously (this has to do with line lengths during presentation).
				# These are cases of reading times > 2000.
				if tot_fix_dur > 2000:
					previous_fixation = tot_fix_dur
					continue
				
				# Remove words with punctuation, i.e., only keep words were WLEN =
				# OLEN. This is because reading times at sentence and phrase boundaries
				# behave differently.
				if row[self.map['OLEN']] != row[self.map['WLEN']]:
					previous_fixation = tot_fix_dur
					continue

				# Remove numbers, i.e., words with PoS tag NUM. These are unreliable,
				# according to Demberg. An more stringent approach is to remove all
				# words whose strings include digits, special symbols (dollar sign etc),
				# or several upper case letters.
				if row[self.map['UniversalPOS']] == 'NUM':
					continue

				# Remove cases with large launch distance values, i.e., LDIST > 20 or
				# LDIST < -30. These indicate beginning or ends of lines, where wrap-up
				# can inflate reading times.
				try:
					ldist = float(row[self.map['LAUN']])
					if ldist > 20 or ldist < -30:
						continue
				except:
					pass
				# Remove cases with missing n-1 or n+1 BNC or Dundee frequencies
				# (first or last word of a text, maybe some other cases).	
				n_plus1 = row[self.map['n+1_Dun_freq']]
				n_minus1 = row[self.map['n-1_Dun_freq']]
				if n_plus1 == '' or n_minus1 == '':
					continue	

				if previous_fixation != -1:
					add_to_res(row, previous_fixation)


			rev_mapping = {v:k for k,v in self.map.items()}
			final = {rev_mapping[j]:result[i] for i,j in enumerate(sorted(self.wantedSet))}
			final['previousFixation'] = result[-1]

			return final
	
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





wantedSet = ['First_pass_dur', 'Mean_fix_dur', 'Tot_fix_dur', 'WLEN', 'WORD', 'WNUM', 'UniversalPOS', 'CPOS']
wantedParsing = {'First_pass_dur':float_or_blank_cast, 'Mean_fix_dur':float_or_blank_cast, 'Tot_fix_dur':float_or_blank_cast, 'WLEN':float, 'WORD':string_lower_nopunct_cast, 'WNUM':float, 'UniversalPOS':str, 'CPOS':str}

csv_path = '/media/fede/fedeProSD/eyewordembed/dataset/dundee_eyemovement/treebank/en_Dundee_DLT_freq_goldtok_gr.csv'
save_dir = '/media/fede/fedeProSD/eyewordembed/dataset/dundee_parsed_gr_new/'

if not os.path.isdir(save_dir):
	os.makedirs(save_dir)

dtp = DundeeTreebankParser()
result = dtp.parseFile(csv_path, wantedSet, wantedParsing)


for k,v in result.items():
	util.save(v, os.path.join(save_dir,k))
dtp.fileToText(csv_path, os.path.join(save_dir,'dundee.txt'))