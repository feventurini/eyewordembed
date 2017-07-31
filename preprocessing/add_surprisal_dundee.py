import csv
import string
import sys
sys.path.insert(0, '../utilities')
import timing
import util
import os
import dundee_parser as dp
import kenlm
import math

if __name__ == '__main__':
	avg = False

	if avg:
		csv_path = '/media/fede/fedeProSD/eyewordembed/dataset/dundee_eyemovement/treebank/en_Dundee_DLT_freq_goldtok_gr.csv'
	else:
		csv_path = '/media/fede/fedeProSD/eyewordembed/dataset/dundee_eyemovement/treebank/en_Dundee_DLT_freq_goldtok.csv'

	final_out = csv_path[:-4] + '_with_surprisal.csv'
	dtp = dp.DundeeTreebankParser()

	for i in range(2,6):
		print('Adding {}gram_surprisal...'.format(i))
		if i!=2:
			csv_path = 'temp{}.csv'.format(i-1)

		if i==5:
			out_path = final_out
		else:
			out_path = 'temp{}.csv'.format(i)

		e = math.exp(1)
		
		with open(csv_path,'r') as f:
			with open(out_path, 'w+') as out:
				out_csv = csv.writer(out, delimiter='\t')
				# for order in range(2,5):
				print('Loading language model...')
				model = kenlm.LanguageModel('../dataset/gigawordlm_{}gram.klm'.format(i))
				print('Loaded language model.')
				r = csv.reader(f, delimiter='\t')
				firstrow = next(r)
				dtp.initiateParser(firstrow)
				firstrow.append('{}gram_surprisal'.format(i))
				out_csv.writerow(firstrow)
				sentence = []
				rows = []
				sentence_num = -1
				for row in r:
					if sentence_num == -1:
						sentence_num = row[dtp.map['SentenceID']]
					if sentence_num != row[dtp.map['SentenceID']]:
						sentence_num = row[dtp.map['SentenceID']]
						# print(sentence)
						scores = list(model.full_scores(' '.join(sentence)))
						for s,ro in zip(scores[:-1], rows):
							ro.append(-(s[0]/math.log(e, 10)))
							#input(ro)
							out_csv.writerow(ro)
						sentence = []
						rows = []

					sentence.append(row[dtp.map['Word']].lower())
					rows.append(row)


