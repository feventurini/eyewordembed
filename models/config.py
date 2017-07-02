from chainer import functions as F

train_tarball = '../gigaword_train.tar.bz2'

# ---------------------------------------------- #
## EYETRACKING PARAMETERS
gpu = -1
model_eyetracking_inference = 'linreg' ## must be in ['linreg', 'context']
out_type_eyetracking = 'id' ## must be in ['tanh', 'sigmoid', 'id', 'relu']
#batchsize_eyetracking = 100
if model_eyetracking_inference == 'context':
    window_eyetracking = 1
reg_coeff = 0.001
lens = True
pos = True
n_pos_units = 50

# ---------------------------------------------- #
## WORD2VEC PARAMETERS
window = 5
batchsize_word2vec = 100000
model_word2vec = 'cbow' ## must be in ['skipgram', 'cbow']
out_type_word2vec = 'ns' ## must be in ['ns', 'hsm', 'original'] 

## GENSIM PARAMETERS
alpha = 0.025
min_count = 5
max_vocab_size = 400000
sub_sampling = 0.001
n_workers = 20
cbow_mean = 1 # 1:mean, 0:sum
report_delay = 3.0

vocab_folder = 'init_vocab'
# ---------------------------------------------- #
## SHARED PARAMETERS
n_units = 100
test = False
out_folder = 'result'
epoch = 20

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
print('Training model eyetracking: {}'.format(model_eyetracking_inference))
print('Output type eyetracking: {}'.format(out_type_eyetracking))
print('')
