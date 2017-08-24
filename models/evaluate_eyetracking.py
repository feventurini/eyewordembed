import sys
sys.path.insert(0, '../utilities')
sys.path.insert(0, '.')

import os
import chainer
from chainer import serializers as S

import prepare_dataset as pd

from eyetracking_batch_iter import EyetrackingBatchIterator
from eyetracking import *
from chainer import functions as F
import gensim


window_eyetracking = 0
n_layers = 0
wlen = True
pos = True
prev_fix = True
freq = True
surprisal = True
out_type_eyetracking = 'id'
n_units = 100
bins = False
gpu = -1

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

if os.path.exists(sys.argv[1] + '.model'):
	model = gensim.models.Word2Vec.load(sys.argv[1] + '.model')
	vocab, pos2id, train, val, test, mean, std = pd.load_dataset(model.wv.vocab, gensim=True)
	n_vocab = len(model.wv.vocab)
	n_pos = len(pos2id)
else:
	vocab, pos2id, train, val, test, mean, std = pd.load_dataset()
	n_vocab = len(vocab)
	n_pos = len(pos2id)

loss_func = F.mean_squared_error

model_eyetracking = EyetrackingLinreg(n_vocab, n_units, loss_func, out_eyetracking, window=window_eyetracking, n_layers=n_layers, wlen=wlen, pos=pos, prev_fix=prev_fix, freq=freq, surprisal=surprisal, n_pos=n_pos)

S.load_npz(sys.argv[1] + '.eyemodel', model_eyetracking)

def r2_score(x, y):
    zx = (x-np.mean(x))/np.std(x, ddof=1)
    zy = (y-np.mean(y))/np.std(y, ddof=1)
    r = np.sum(zx*zy)/(len(x)-1)
    return r**2        

test_iter = EyetrackingBatchIterator(val, window_eyetracking, len(test), repeat=False, shuffle=True, wlen=wlen, pos=pos, prev_fix=prev_fix, freq=freq, surprisal=surprisal, bins=bins)
test_set = list(test_iter.next())
for t in test_iter:
    x, y = t
    for i in x:
        test_set[0][i] = np.concatenate((test_set[0][i],x[i]), axis=0)
    test_set[1] = np.concatenate((test_set[1],y), axis=0)

test_set = convert(tuple(test_set), gpu)
inputs, target = test_set
predictions = model_eyetracking.inference(inputs)
mse = F.mean_squared_error(predictions,target)
# target = std*target + mean
# predictions = std*predictions + mean
# for t, i in zip(target, predictions):
#     print(t, i)
r2 = r2_score(target, predictions)
print('R_squared coefficient: {}'.format(r2))
print('Mean squared error: {}'.format(mse.data))
# with open(out_folder + os.sep + '{}.r2'.format(name), 'w+') as out:
#     print('R_squared coefficient: {}'.format(r2), file=out)

