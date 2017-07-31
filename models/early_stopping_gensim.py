from chainer.training import extension

def early_stopping_gensim(model_word2vec, filename):
    @extension.make_extension(trigger=(1, 'epoch'), priority=-100)
    def snapshot_object(trainer):
        model_word2vec.save(filename)

    return snapshot_object