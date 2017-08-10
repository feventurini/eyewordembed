import chainer
import chainer.functions as F
import chainer.initializers as I
import chainer.links as L
import chainer.optimizers as O
from chainer import reporter
import numpy as np

def toOneHot(n, n_participants):
    res = np.eye(n_participants, dtype=np.float32)[n]
    return res

class BaselineClassifier(chainer.Chain):
    def __init__(self, mean, loss_func):
        super(Baseline, self).__init__()
        self.loss_func = loss_func
        self.mean = mean

    def __call__(self, inputs, target):
        o = np.full(target.shape, self.mean, dtype=np.float32)
        loss = self.loss_func(o, target)
        acc = F.accuracy(F.softmax(o), target)
        reporter.report({'loss': loss}, self)
        reporter.report({'accuracy':acc}, self)
        return loss        

class EyetrackingClassifier(chainer.Chain):

    def __init__(self, n_vocab, n_units, n_participants, n_classes, loss_func, out, n_layers=0, window=0, wlen=False, 
        pos=False, prev_fix=False, freq=False, surprisal=False, n_pos=None, n_hidden=200, n_pos_units=50, loss_ratio=1.0):
        super(EyetrackingClassifier, self).__init__()

        self.n_units = n_units
        self.n_pos_units = n_pos_units
        self.pos = pos
        self.wlen = wlen
        self.prev_fix = prev_fix
        self.freq = freq
        self.surprisal = surprisal
        self.loss_ratio = loss_ratio
        self.n_participants = n_participants
        self.n_layers = n_layers

        with self.init_scope():
            self.embed = L.EmbedID(n_vocab, n_units, initialW=I.Uniform(1. / n_units))

            n_inputs = n_units
            if self.pos: 
                assert(n_pos)
                self.embed_pos = L.EmbedID(n_pos, n_pos_units, initialW=I.Uniform(1. / n_pos_units))
                n_inputs += n_pos_units

            if self.prev_fix:
                n_inputs += n_participants
            
            if self.wlen:
                n_inputs += 1

            if self.freq:
                n_inputs += 1

            if self.surprisal:
                n_inputs += 1
            
            n_inputs *= (window + 1)
            
            if n_layers > 0:
                self.layer0 = L.Linear(n_inputs, n_hidden, initialW=I.Uniform(1. / n_hidden))
                for i in range(1, n_layers):
                    setattr(self, 'layer{}'.format(i), L.Linear(n_hidden, n_hidden, initialW=I.Uniform(1. / n_hidden)))
                self.outlayer = L.Linear(n_hidden, n_classes, initialW=I.Uniform(1. / n_hidden))
            else:
                self.outlayer = L.Linear(n_inputs, n_classes, initialW=I.Uniform(1. / n_inputs))
 
            self.out = out
            self.loss_func = loss_func

    def _embed_input(self, inputs):
        variables = []

        w = chainer.Variable(inputs['words'], name='words')
        e_w = F.reshape(self.embed(w), (-1,w.shape[1]*self.n_units))
        variables.append(e_w)

        if self.pos:
            p = chainer.Variable(inputs['pos'], name='pos_tags')
            e_p = F.reshape(self.embed_pos(p), (-1,w.shape[1]*self.n_pos_units))
            e_p.name = 'pos_embeddings'
            variables.append(e_p)

        if self.wlen:
            l = chainer.Variable(inputs['wlen'], name='word_lengths')
            variables.append(l)

        if self.prev_fix:
            t = chainer.Variable(toOneHot(inputs['prev_fix'], self.n_participants), name='previous_fixations')
            t = F.reshape(t, (-1, w.shape[1]*self.n_participants))
            t.name = 'previous_fixations'
            variables.append(t)

        if self.freq:
            f = chainer.Variable(inputs['freq'], name='frequency')
            variables.append(f)

        if self.surprisal:
            s = chainer.Variable(inputs['surprisal'], name='surprisal')
            variables.append(s)

        h = F.concat(tuple(variables), axis=1)# * (1. / w.shape[1])

        h.name = 'concatenated_word_embeddings'
        return h

    def __call__(self, inputs, target):
        target = chainer.Variable(target, name='target')
        h = self._embed_input(inputs) # called from superclass
        for i in range(self.n_layers):
            h = self.out(getattr(self,'layer{}'.format(i))(h))
        o = self.outlayer(h)
        o.name = 'output_time_prediction'

        loss = self.loss_func(o, target)
        acc = F.accuracy(F.softmax(o), target)
        reporter.report({'loss': loss}, self)
        reporter.report({'accuracy':acc}, self)
        return self.loss_ratio * loss

    def inference(self, inputs):
        h = self._embed_input(inputs) # called from superclass
        for i in range(self.n_layers):
            h = self.out(getattr(self,'layer{}'.format(i))(h))
        o = self.outlayer(h)
        return F.softmax(o)


class BaselineLinreg(chainer.Chain):
    def __init__(self, mean, loss_func):
        super(Baseline, self).__init__()
        self.loss_func = loss_func
        self.mean = mean

    def __call__(self, inputs, target):
        loss = self.loss_func(np.full(target.shape, self.mean, dtype=np.float32), target)
        reporter.report({'loss': loss}, self)
        return loss        

class EyetrackingLinreg(chainer.Chain):

    def __init__(self, n_vocab, n_units, loss_func, out, window=0, n_layers=0, n_hidden=200, wlen=False, 
        pos=False, prev_fix=False, freq=False, surprisal=False, n_pos=None, n_pos_units=50, loss_ratio=1.0):
        super(EyetrackingLinreg, self).__init__()

        self.n_units = n_units
        self.n_pos_units = n_pos_units
        self.pos = pos
        self.wlen = wlen
        self.prev_fix = prev_fix
        self.freq = freq
        self.surprisal = surprisal
        self.loss_ratio = loss_ratio
        self.n_layers = n_layers
        self.n_pos = n_pos

        with self.init_scope():
            self.embed = L.EmbedID(n_vocab, n_units, initialW=I.Uniform(1. / n_units))

            n_inputs = n_units
            if self.pos: 
                assert(n_pos)
                ## embedding
                # self.embed_pos = L.EmbedID(n_pos, n_pos_units, initialW=I.Uniform(1. / n_pos_units))
                # n_inputs += n_pos_units
                
                ## one-hot
                n_inputs += n_pos

            if self.prev_fix:
                n_inputs += 1
            
            if self.wlen:
                n_inputs += 1

            if self.freq:
                n_inputs += 1

            if self.surprisal:
                n_inputs += 1
            
            n_inputs *= (window + 1)

            if n_layers > 0:
                self.layer0 = L.Linear(n_inputs, n_hidden, initialW=I.Uniform(1. / n_hidden))
                for i in range(1, n_layers):
                    setattr(self, 'layer{}'.format(i), L.Linear(n_hidden, n_hidden, initialW=I.Uniform(1. / n_hidden)))
                self.outlayer = L.Linear(n_hidden, 1, initialW=I.Uniform(1. / n_hidden))
            else:
                self.outlayer = L.Linear(n_inputs, 1, initialW=I.Uniform(1. / n_inputs))
 
            self.out = out
            self.loss_func = loss_func

    def _embed_input(self, inputs):
        variables = []

        w = chainer.Variable(inputs['words'], name='words')
        e_w = F.reshape(self.embed(w), (-1,w.shape[1]*self.n_units))
        variables.append(e_w)

        if self.pos:
            ## embedding
            # p = chainer.Variable(inputs['pos'], name='pos_tags')
            # e_p = F.reshape(self.embed_pos(p), (-1,w.shape[1]*self.n_pos_units))
            # e_p.name = 'pos_embeddings'
            
            ## one-hot
            p = chainer.Variable(toOneHot(inputs['pos'],self.n_pos), name='pos_tags')
            e_p = F.reshape(p, (-1,w.shape[1]*self.n_pos))
            e_p.name = 'postags_onehot'
            variables.append(e_p)

        if self.wlen:
            l = chainer.Variable(inputs['wlen'], name='word_lengths')
            variables.append(l)

        if self.prev_fix:
            t = chainer.Variable(inputs['prev_fix'], name='previous_fixations')
            t.name = 'previous_fixations'
            variables.append(t)

        if self.freq:
            f = chainer.Variable(inputs['freq'], name='frequency')
            variables.append(f)

        if self.surprisal:
            s = chainer.Variable(inputs['surprisal'], name='surprisal')
            variables.append(s)

        h = F.concat(tuple(variables), axis=1)# * (1. / w.shape[1])

        h.name = 'concatenated_word_embeddings'
        return h

    def __call__(self, inputs, target):
        target = chainer.Variable(target, name='target')
        h = self._embed_input(inputs) # called from superclass
        for i in range(self.n_layers):
            h = self.out(getattr(self,'layer{}'.format(i))(h))
        o = self.out(self.outlayer(h))
        o.name = 'output_time_prediction'

        loss = self.loss_func(o, target)
        reporter.report({'loss': loss}, self)
        return self.loss_ratio * loss

    def inference(self, inputs):
        h = self._embed_input(inputs) # called from superclass
        for i in range(self.n_layers):
            h = self.out(getattr(self,'layer{}'.format(i))(h))
        o = self.out(self.outlayer(h))
        return o.data      

def convert(batch, device):
    x, targets = batch
    for k in x:
        if device >= 0:
            x[k] = cuda.to_gpu(x[k])
    if device >= 0:
        targets = cuda.to_gpu(targets)
    return x, targets