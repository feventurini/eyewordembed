import gensim
import sys

sys.path.insert(0, '../utilities')
import util


if sys.argv[2] == 'lm':
	multitask = '../models/test_multitask_limit_vocab/result/tokenized_gigaword_4/skipgram/model_skipgram_linreg_0window_0layers_id_adagrad_0.01lr_0.001reg_coeff_1.0lossratio_4downsample.model'
	std = '../models/test_multitask_limit_vocab/result_final/tokenized_gigaword_4/skipgram/std_limit_vocab_word2vec.model'
else:
	multitask = '../models/test_multitask/result/tokenized_gigaword_4/skipgram/model_skipgram_linreg_0window_0layers_id_adagrad_0.01lr_0.001reg_coeff_1.0lossratio_4downsample.model'
	std = '../models/test_multitask/result_final/tokenized_gigaword_4/skipgram/std_word2vec.model'

temp = {}
words = util.load('../dataset/dundee_parsed_gr/Word')
times = util.load('../dataset/dundee_parsed_gr/Tot_fix_dur')

for w,t in zip(words, times):
	if w not in temp:
		temp[w] = (t,1)
	else:
		temp[w] = (temp[w][0] + t, temp[w][1] + 1)

avg_times = {w:i[0]/i[1] for w,i in temp.items()}

model_multi = gensim.models.Word2Vec.load(multitask)
model_std = gensim.models.Word2Vec.load(std) 

word = sys.argv[1]


print('{}\t{}'.format(word, avg_times[word]))
# print('Multitask...')
for w,t in model_multi.most_similar(word, topn=5):
	print('{}\t{}'.format(w, avg_times[w] if w in avg_times else '-'))
print()
# print('Standard...')
for w,t in model_std.most_similar(word, topn=5):
	print('{}\t{}'.format(w, avg_times[w] if w in avg_times else '-'))
