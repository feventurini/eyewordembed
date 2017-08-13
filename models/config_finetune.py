from chainer import functions as F

train_tarball = '../dataset/downsampled_gigaword/tokenized_gigaword_8.tar.bz2'

# ---------------------------------------------- #
## EYETRACKING PARAMETERS
gpu = -1
out_type_eyetracking = 'id' ## must be in ['tanh', 'sigmoid', 'id', 'relu']
window_eyetracking = 0
reg_coeff = 0.0
wlen = True
pos = True
prev_fix = True
freq = True
n_pos_units = 50
n_hidden = 200
n_layers = 0
bins = False

# ---------------------------------------------- #
## WORD2VEC PARAMETERS
window = 5
model_word2vec = 'skipgram' ## must be in ['skipgram', 'cbow']
out_type_word2vec = 'ns' ## must be in ['ns', 'hsm', 'original'] 

## GENSIM PARAMETERS
alpha = 0.025
min_count = 5
max_vocab_size = 400000
sub_sampling = 0.001
n_workers = 3
cbow_mean = 1 # 1:mean, 0:sum
report_delay = 3.0

vocab_folder = 'init_vocab'

# ---------------------------------------------- #
## SHARED PARAMETERS
n_units = 300
test = False
out_folder = 'result'
epoch = 10
epoch_ratio = 1.0
loss_ratio = 1.0
batchsize_eyetracking = 1000

if out_type_eyetracking == 'tanh':
    out_eyetracking = F.tanh
elif out_type_eyetracking == 'relu':
    out_eyetracking = F.relu
elif out_type_eyetracking == 'sigmoid':
    out_eyetracking = F.sigmoid
elif out_type_eyetracking == 'id':
    out_eyetracking = F.identity
else:
    raise Exception('Unknown output type: {}'.format(out_type))

if out_type_word2vec == 'hsm':
	hs = 1
	negative = 0
elif out_type_word2vec == 'ns':
	hs = 0
	negative = 5
elif out_type_word2vec == 'original':
	hs = 0
	negative = 0
else:
    raise Exception('Unknown output type: {}'.format(out_type))

if model_word2vec == 'skipgram':
	sg = 1
elif model_word2vec == 'cbow':
	sg = 0
else:
    raise Exception('Unknown model type: {}'.format(model))


print('GPU: {}'.format(gpu))
print('# unit: {}'.format(n_units))
print('# epoch: {}'.format(epoch))
print('Window word2vec: {}'.format(window))
#print('Minibatch-size word2vec: {}'.format(batchsize_word2vec))
print('Training model word2vec: {}'.format(model_word2vec))
print('Output type word2vec: {}'.format(out_type_word2vec))
print('')
#print('Minibatch-size eyetracking: {}'.format(batchsize_eyetracking))
print('Training model eyetracking: {}'.format('with bins' if bins else 'without bins'))
print('Output type eyetracking: {}'.format(out_type_eyetracking))
print('')
