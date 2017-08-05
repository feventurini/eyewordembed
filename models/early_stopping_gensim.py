from chainer.training import extension
from chainer import serializers as S

def early_stopping_gensim(model_word2vec, model_eyetracking, filename):
    @extension.make_extension(trigger=(1, 'epoch'), priority=-100)
    def snapshot_object(trainer):
        model_word2vec.save('{}.model'.format(filename))
        S.save_npz('{}.eyemodel'.format(filename), model_eyetracking)        

    return snapshot_object