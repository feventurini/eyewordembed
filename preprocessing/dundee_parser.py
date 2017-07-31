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
		#print(firstRow)
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

			def add_to_res(row):
				temp = dtp.parserow(row)
				for i in range(len(temp)):
					result[i].append(temp[i])

			index_tot_fix_dur = [i for i,j in enumerate(sorted(self.wantedSet)) if j==self.map['Tot_fix_dur']][0]
			count_zero = 0

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
					continue
				
				# Remove words with punctuation, i.e., only keep words were WLEN =
				# OLEN. This is because reading times at sentence and phrase boundaries
				# behave differently.
				if row[self.map['OLEN']] != row[self.map['WLEN']]:
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

				add_to_res(row)


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





if __name__ == '__main__':
	avg = True

	if avg:
		wantedSet = ['First_pass_dur', 'First_fix_dur', 'Mean_fix_dur', 'Tot_fix_dur', 
					'WLEN', 'WORD', 'WNUM', 'UniversalPOS', 'CPOS', 'BNC_freq', 
					'n-1_fix_dur', '2gram_surprisal', '3gram_surprisal', '4gram_surprisal', '5gram_surprisal']
		wantedParsing = {'First_pass_dur':float_or_blank_cast, 'Mean_fix_dur':float_or_blank_cast, 
						'Tot_fix_dur':float_or_blank_cast, 'WLEN':float, 'WORD':string_lower_nopunct_cast, 
						'WNUM':float, 'UniversalPOS':str, 'CPOS':str, 'BNC_freq':float_or_blank_cast, 
						'First_fix_dur':float_or_blank_cast, 'n-1_fix_dur':float_or_blank_cast,
						'2gram_surprisal':float_or_blank_cast, '3gram_surprisal':float_or_blank_cast,
						'4gram_surprisal':float_or_blank_cast, '5gram_surprisal':float_or_blank_cast,}
		csv_path = '/media/fede/fedeProSD/eyewordembed/dataset/dundee_eyemovement/treebank/en_Dundee_DLT_freq_goldtok_gr_with_surprisal.csv'
		save_dir = '/media/fede/fedeProSD/eyewordembed/dataset/dundee_parsed_gr/'

	else:
		wantedSet = ['First_pass_dur', 'First_fix_dur', 'Mean_fix_dur', 'Tot_fix_dur', 
					'WLEN', 'WORD', 'WNUM', 'UniversalPOS', 'CPOS', 'BNC_freq', 
					'n-1_fix_dur', '2gram_surprisal', '3gram_surprisal', '4gram_surprisal', '5gram_surprisal', 
					'Participant']
		wantedParsing = {'First_pass_dur':float_or_blank_cast, 'Mean_fix_dur':float_or_blank_cast, 
						'Tot_fix_dur':float_or_blank_cast, 'WLEN':float, 'WORD':string_lower_nopunct_cast, 
						'WNUM':float, 'UniversalPOS':str, 'CPOS':str, 'First_fix_dur':float_or_blank_cast, 
						'BNC_freq':float_or_blank_cast, 'n-1_fix_dur':float_or_blank_cast, 
						'2gram_surprisal':float_or_blank_cast, '3gram_surprisal':float_or_blank_cast,
						'4gram_surprisal':float_or_blank_cast, '5gram_surprisal':float_or_blank_cast,
						'Participant':str}
		csv_path = '/media/fede/fedeProSD/eyewordembed/dataset/dundee_eyemovement/treebank/en_Dundee_DLT_freq_goldtok_with_surprisal.csv'
		save_dir = '/media/fede/fedeProSD/eyewordembed/dataset/dundee_parsed/'

	if not os.path.isdir(save_dir):
		os.makedirs(save_dir)

	dtp = DundeeTreebankParser()
	result = dtp.parseFile(csv_path, wantedSet, wantedParsing)


	for k,v in result.items():
		util.save(v, os.path.join(save_dir,k))
	dtp.fileToText(csv_path, os.path.join(save_dir,'dundee.txt'))

