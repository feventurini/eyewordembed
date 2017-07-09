import chainer
import chainer.functions as F
import chainer.initializers as I
import chainer.links as L
import chainer.optimizers as O
from chainer import reporter

class Baseline(chainer.Chain):
    def __init__(self, mean, loss_func):
        super(Baseline, self).__init__()
        self.loss_func = loss_func
        self.mean = mean

    def __call__(self, inputs, target):
        loss = self.loss_func(np.full(target.shape, self.mean, dtype=np.float32), target)
        reporter.report({'loss': loss}, self)
        return loss        

class LinReg(chainer.Chain):

    def __init__(self, n_vocab, n_units, loss_func, out, wlen=False, pos=False, prev_time=False, n_pos=None, n_pos_units=50):
        super(LinReg, self).__init__()

        self.pos = pos
        self.wlen = wlen
        self.prev_time = prev_time

        with self.init_scope():
            self.embed = L.EmbedID(n_vocab, n_units, initialW=I.Uniform(1. / n_units))

            if self.pos and self.wlen and self.prev_time:
                assert(n_pos)
                self.embed_pos = L.EmbedID(n_pos, n_pos_units, initialW=I.Uniform(1. / n_pos_units))
                n_inputs = n_units + n_pos_units + 2

            elif self.pos and self.prev_time:
                assert(n_pos)
                self.embed_pos = L.EmbedID(n_pos, n_pos_units, initialW=I.Uniform(1. / n_pos_units))
                n_inputs = n_units + n_pos_units + 1 
            
            elif self.wlen and self.prev_time:
                self.embed_pos = L.EmbedID(n_pos, n_pos_units, initialW=I.Uniform(1. / n_pos_units))
                n_inputs = n_units + 2 
            
            elif self.pos and self.wlen:
                assert(n_pos)
                self.embed_pos = L.EmbedID(n_pos, n_pos_units, initialW=I.Uniform(1. / n_pos_units))
                n_inputs = n_units + n_pos_units + 1 
            
            elif self.wlen:
                n_inputs = n_units + 1
            
            elif self.prev_time:
                n_inputs = n_units + 1
            
            elif self.pos:
                assert(n_pos)
                self.embed_pos = L.EmbedID(n_pos, n_pos_units, initialW=I.Uniform(1. / n_pos_units))
                n_inputs = n_units + n_pos_units
            
            else:
                n_inputs = n_units

            self.lin = L.Linear(n_inputs, 1, initialW=I.Uniform(1. / n_inputs))            
            self.out = out
            self.loss_func = loss_func

    def _embed_input(self, inputs):
        if self.pos and self.wlen and self.prev_time:
            w, l, p, t = inputs
            w = chainer.Variable(w, name='word')
            p = chainer.Variable(p, name='pos_tag')
            l = chainer.Variable(l, name='word_length')
            t = chainer.Variable(t, name='previous_fixation')

            e_w = self.embed(w)
            e_p = self.embed_pos(p)
            l = F.reshape(l,(-1,1,1))
            t = F.reshape(t,(-1,1,1))
            h = F.concat((e_w, e_p, l, t), axis=2)

            t.name = 'previous_fixation'
            l.name = 'word_length'
            e_p.name = 'pos_embedding'

        elif self.pos and self.wlen:
            w, l, p = inputs
            w = chainer.Variable(w, name='word')
            p = chainer.Variable(p, name='pos_tag')
            l = chainer.Variable(l, name='word_length')

            e_w = self.embed(w)
            e_p = self.embed_pos(p)
            l = F.reshape(l,(-1,1,1))
            h = F.concat((e_w, e_p, l), axis=2)

            l.name = 'word_length'
            e_p.name = 'pos_embedding'

        elif self.pos and self.prev_time:
            w, l, p = inputs
            w = chainer.Variable(w, name='word')
            p = chainer.Variable(p, name='pos_tag')
            t = chainer.Variable(t, name='previous_fixation')

            e_w = self.embed(w)
            e_p = self.embed_pos(p)
            t = F.reshape(t,(-1,1,1))
            h = F.concat((e_w, e_p, t), axis=2)

            t.name = 'previous_fixation'
            e_p.name = 'pos_embedding'

        elif self.prev_time and self.wlen:
            w, l, t = inputs
            w = chainer.Variable(w, name='word')
            l = chainer.Variable(l, name='word_length')
            t = chainer.Variable(t, name='previous_fixation')

            e_w = self.embed(w)
            l = F.reshape(l,(-1,1,1))
            t = F.reshape(t,(-1,1,1))
            h = F.concat((e_w, l, t), axis=2)

            l.name = 'word_length'
            t.name = 'previous_fixation'

        elif self.pos:
            w, p = inputs
            w = chainer.Variable(w, name='word')
            p = chainer.Variable(p, name='pos_tag')

            e_w = self.embed(w)
            e_p = self.embed_pos(p)
            h = F.concat((e_w, e_p), axis=2)

            e_p.name = 'pos_embedding'

        elif self.wlen:
            w, l = inputs

            w = chainer.Variable(w, name='word')
            l = chainer.Variable(l, name='word_length')

            e_w = self.embed(w)
            l = F.reshape(l,(-1,1,1))
            h = F.concat((e_w, l), axis=2)

            l.name = 'word_length'

        elif self.prev_time:
            w, t = inputs

            w = chainer.Variable(w, name='word')
            t = chainer.Variable(t, name='previous_fixation')

            e_w = self.embed(w)
            t = F.reshape(t,(-1,1,1))
            h = F.concat((e_w, t), axis=2)

            l.name = 'word_length'

        else:
            if isinstance(inputs, tuple):
                w = inputs[0]
            else:
                w = inputs
            h = self.embed(w)

        h.name = 'word_embedding'
        return h

    def __call__(self, inputs, target):
        target = chainer.Variable(target, name='target')
        h = self._embed_input(inputs)
        o = self.out(self.lin(h))
        o.name = 'output_time_prediction'
        loss = self.loss_func(o, target)
        reporter.report({'loss': loss}, self)
        return loss

    def inference(self, inputs):
        h = self._embed_input(inputs)
        return self.out(self.lin(h))

class Multilayer(LinReg):

    def __init__(self, n_vocab, n_units, loss_func, out, n_hidden=50, n_layers=1, wlen=False, pos=False, prev_time=False, n_pos=None, n_pos_units=50):
        super(LinReg, self).__init__()

        self.pos = pos
        self.wlen = wlen
        self.prev_time = prev_time

        with self.init_scope():
            self.embed = L.EmbedID(n_vocab, n_units, initialW=I.Uniform(1. / n_units))

            if self.pos and self.wlen and self.prev_time:
                assert(n_pos)
                self.embed_pos = L.EmbedID(n_pos, n_pos_units, initialW=I.Uniform(1. / n_pos_units))
                n_inputs = n_units + n_pos_units + 2

            elif self.pos and self.prev_time:
                assert(n_pos)
                self.embed_pos = L.EmbedID(n_pos, n_pos_units, initialW=I.Uniform(1. / n_pos_units))
                n_inputs = n_units + n_pos_units + 1 
            
            elif self.wlen and self.prev_time:
                self.embed_pos = L.EmbedID(n_pos, n_pos_units, initialW=I.Uniform(1. / n_pos_units))
                n_inputs = n_units + 2 
            
            elif self.pos and self.wlen:
                assert(n_pos)
                self.embed_pos = L.EmbedID(n_pos, n_pos_units, initialW=I.Uniform(1. / n_pos_units))
                n_inputs = n_units + n_pos_units + 1 
            
            elif self.wlen:
                n_inputs = n_units + 1
            
            elif self.prev_time:
                n_inputs = n_units + 1
            
            elif self.pos:
                assert(n_pos)
                self.embed_pos = L.EmbedID(n_pos, n_pos_units, initialW=I.Uniform(1. / n_pos_units))
                n_inputs = n_units + n_pos_units
            
            else:
                n_inputs = n_units

            self.lin = L.Linear(n_inputs, n_hidden, initialW=I.Uniform(1. / n_inputs))
            
            self.layers = list()
            for i in range(n_layers - 1):
                self.layers.append(L.Linear(n_hidden, n_hidden, initialW=I.Uniform(1. / n_units)))
            self.layers.append(L.Linear(n_hidden, 1, initialW=I.Uniform(1. / n_units)))
            
            self.out = out
            self.loss_func = loss_func

    def __call__(self, inputs, target):
        target = chainer.Variable(target, name='target')
        i = (self._embed_input(inputs)) # called from superclass
        h = self.out(self.lin(i))
        for l in self.layers:
            h = self.out(l(h))

        h.name = 'output_time_prediction'
        loss = self.loss_func(h, target)
        reporter.report({'loss': loss}, self)
        return loss

    def inference(self, inputs):
        i = (self._embed_input(inputs)) # called from superclass
        h = self.out(self.lin(i))
        for l in self.layers:
            h = self.out(l(h))
        return h

@DeprecationWarning
class LinRegContextSum(chainer.Chain):

    def __init__(self, n_vocab, n_units, loss_func, out, window=1, wlen=False, pos=False, prev_time=False, n_pos=None, n_pos_units=50):
        super(LinRegContextSum, self).__init__()

        self.pos = pos
        self.wlen = wlen

        with self.init_scope():
            self.embed = L.EmbedID(n_vocab, n_units, initialW=I.Uniform(1. / n_units))

            if self.pos and self.wlen:
                assert(n_pos)
                n_inputs = n_units + n_pos_units + 1
                self.embed_pos = L.EmbedID(n_pos, n_pos_units, initialW=I.Uniform(1. / n_pos_units))

            elif self.wlen:
                n_inputs = n_units + 1
            
            elif self.pos:
                n_inputs = n_units + n_pos_units
                self.embed_pos = L.EmbedID(n_pos, n_pos_units, initialW=I.Uniform(1. / n_pos_units))
            
            else:
                n_inputs = n_units

            self.lin = L.Linear(n_inputs, 1, initialW=I.Uniform(1. / n_inputs))
            self.out = out
            self.loss_func = loss_func

    def _embed_input(self, inputs):
        if self.pos and self.wlen:
            w, l, p = inputs

            w = chainer.Variable(w, name='words_window')
            p = chainer.Variable(p, name='pos_tags_window')
            l = chainer.Variable(l, name='word_lengths_window')

            e_w = F.sum(self.embed(w), axis=1) * (1. / w.shape[1])
            e_p = F.sum(self.embed_pos(p), axis=1) * (1. / p.shape[1])
            l = F.mean(l, axis=1).reshape(-1,1) #F.reshape(l,(-1,w.shape[1]))
            h = F.concat((e_w, e_p, l), axis=1)

            l.name = 'word_lengths_window'
            e_p.name = 'pos_embeddings'

        elif self.pos:
            w, p = inputs

            w = chainer.Variable(w, name='words_window')
            p = chainer.Variable(p, name='pos_tags_window')

            e_w = F.sum(self.embed(w), axis=1) * (1. / w.shape[1])
            e_p = F.sum(self.embed_pos(p), axis=1) * (1. / p.shape[1])
            h = F.concat((e_w, e_p), axis=1)

            e_p.name = 'pos_embeddings'

        elif self.wlen:
            w, l = inputs

            w = chainer.Variable(w, name='words_window')
            l = chainer.Variable(l, name='word_lengths_window')

            e_w = F.sum(self.embed(w), axis=1) * (1. / w.shape[1])
            l = F.mean(l, axis=1).reshape(-1,1) #F.reshape(l,(-1,w.shape[1]))
            h = F.concat((e_w, l), axis=1)

            l.name = 'word_lengths_window'

        else:
            if isinstance(inputs, tuple):
                w = inputs[0]
            else:
                w = inputs
            h = F.sum(self.embed(w), axis=1) * (1. / w.shape[1])

        h.name = 'word_embeddings'

        return h

    def __call__(self, inputs, target):
        target = chainer.Variable(target, name='target')
        h = self._embed_input(inputs)
        o = self.out(self.lin(h))
        o.name = 'output_time_prediction'
        loss = self.loss_func(o, target)
        reporter.report({'loss': loss}, self)
        return loss

    def inference(self, inputs):
        h = self._embed_input(inputs)
        return self.out(self.lin(h))

class LinRegContextConcat(chainer.Chain):

    def __init__(self, n_vocab, n_units, loss_func, out, window=1, wlen=False, pos=False, prev_time=False, n_pos=None, n_pos_units=50):
        super(LinRegContextConcat, self).__init__()

        self.n_units = n_units
        self.n_pos_units = n_pos_units
        self.pos = pos
        self.wlen = wlen
        self.prev_time = prev_time

        with self.init_scope():
            self.embed = L.EmbedID(n_vocab, n_units, initialW=I.Uniform(1. / n_units))

            if self.pos and self.wlen and self.prev_time:
                assert(n_pos)
                self.embed_pos = L.EmbedID(n_pos, n_pos_units, initialW=I.Uniform(1. / n_pos_units))
                n_inputs = n_units + n_pos_units + 2

            elif self.pos and self.prev_time:
                assert(n_pos)
                self.embed_pos = L.EmbedID(n_pos, n_pos_units, initialW=I.Uniform(1. / n_pos_units))
                n_inputs = n_units + n_pos_units + 1 
            
            elif self.wlen and self.prev_time:
                self.embed_pos = L.EmbedID(n_pos, n_pos_units, initialW=I.Uniform(1. / n_pos_units))
                n_inputs = n_units + 2 
            
            elif self.pos and self.wlen:
                assert(n_pos)
                self.embed_pos = L.EmbedID(n_pos, n_pos_units, initialW=I.Uniform(1. / n_pos_units))
                n_inputs = n_units + n_pos_units + 1 
            
            elif self.wlen:
                n_inputs = n_units + 1
            
            elif self.prev_time:
                n_inputs = n_units + 1
            
            elif self.pos:
                assert(n_pos)
                self.embed_pos = L.EmbedID(n_pos, n_pos_units, initialW=I.Uniform(1. / n_pos_units))
                n_inputs = n_units + n_pos_units
            
            else:
                n_inputs = n_units

            self.lin = L.Linear(n_inputs * (window + 1), 1, initialW=I.Uniform(1. / n_inputs))
            self.out = out
            self.loss_func = loss_func

    def _embed_input(self, inputs):
        if self.pos and self.wlen and self.prev_time:
            w, l, p, t = inputs
            w = chainer.Variable(w, name='words_window')
            p = chainer.Variable(p, name='pos_tags_window')
            l = chainer.Variable(l, name='word_lengths_window')
            t = chainer.Variable(t, name='previous_fixations_window')

            e_w = F.reshape(self.embed(w), (-1,w.shape[1]*self.n_units))
            e_p = F.reshape(self.embed_pos(p), (-1,w.shape[1]*self.n_pos_units))
            h = F.concat((e_w, e_p, l, t), axis=1)# * (1. / w.shape[1])

            l.name = 'word_lengths_window'
            e_p.name = 'pos_embeddings'
            t.name = 'previous_fixations_window'

        elif self.pos and self.wlen:
            w, l, p = inputs
            w = chainer.Variable(w, name='words_window')
            p = chainer.Variable(p, name='pos_tags_window')
            l = chainer.Variable(l, name='word_lengths_window')

            e_w = F.reshape(self.embed(w), (-1,w.shape[1]*self.n_units))
            e_p = F.reshape(self.embed_pos(p), (-1,w.shape[1]*self.n_pos_units))
            h = F.concat((e_w, e_p, l), axis=1)# * (1. / w.shape[1])

            l.name = 'word_lengths_window'
            e_p.name = 'pos_embeddings'

        elif self.pos and self.prev_time:
            w, l, p = inputs
            w = chainer.Variable(w, name='words_window')
            p = chainer.Variable(p, name='pos_tags_window')
            t = chainer.Variable(t, name='previous_fixations_window')

            e_w = F.reshape(self.embed(w), (-1,w.shape[1]*self.n_units))
            e_p = F.reshape(self.embed_pos(p), (-1,w.shape[1]*self.n_pos_units))
            h = F.concat((e_w, e_p, t), axis=1)# * (1. / w.shape[1])

            t.name = 'previous_fixations_window'
            e_p.name = 'pos_embeddings'

        elif self.wlen and self.prev_time:
            w, l, p = inputs
            w = chainer.Variable(w, name='words_window')
            t = chainer.Variable(t, name='previous_fixations_window')
            l = chainer.Variable(l, name='word_lengths_window')

            e_w = F.reshape(self.embed(w), (-1,w.shape[1]*self.n_units))
            h = F.concat((e_w, l, t), axis=1)# * (1. / w.shape[1])

            t.name = 'previous_fixations_window'
            l.name = 'word_lengths_window'

        elif self.pos:
            w, p = inputs
            w = chainer.Variable(w, name='words_window')
            p = chainer.Variable(p, name='pos_tags_window')

            e_w = F.reshape(self.embed(w), (-1,w.shape[1]*self.n_units))
            e_p = F.reshape(self.embed_pos(p), (-1,w.shape[1]*self.n_pos_units))
            h = F.concat((e_w, e_p), axis=1)

            e_p.name = 'pos_embeddings'

        elif self.wlen:
            w, l = inputs
            w = chainer.Variable(w, name='words_window')
            l = chainer.Variable(l, name='word_lengths_window')

            e_w = F.reshape(self.embed(w), (-1,w.shape[1]*self.n_units))
            h = F.concat((e_w, l), axis=1)

            l.name = 'word_lengths_window'

        elif self.prev_time:
            w, l = inputs
            w = chainer.Variable(w, name='words_window')
            t = chainer.Variable(t, name='previous_fixations_window')

            e_w = F.reshape(self.embed(w), (-1,w.shape[1]*self.n_units))
            h = F.concat((e_w, t), axis=1)

            t.name = 'previous_fixations_window'

        else:
            if isinstance(inputs, tuple):
                w = inputs[0]
            else:
                w = inputs

            w = chainer.Variable(w, name='word')
            h = F.reshape(self.embed(w), (-1,w.shape[1]*self.n_units))

        h.name = 'concatenated_word_embeddings'
        return h

    def __call__(self, inputs, target):
        target = chainer.Variable(target, name='target')
        h = self._embed_input(inputs)
        o = self.out(self.lin(h))
        o.name = 'output_time_prediction'
        loss = self.loss_func(o, target)
        reporter.report({'loss': loss}, self)
        return loss

    def inference(self, inputs):
        h = self._embed_input(inputs)
        return self.out(self.lin(h))

class MultilayerContext(LinRegContextConcat):

    def __init__(self, n_vocab, n_units, loss_func, out, window=1, n_layers=1, n_hidden=50, wlen=False, pos=False, prev_time=False, n_pos=None, n_pos_units=50):
        super(LinRegContextConcat, self).__init__()

        self.n_units = n_units
        self.n_pos_units = n_pos_units
        self.pos = pos
        self.wlen = wlen
        self.prev_time = prev_time

        with self.init_scope():
            self.embed = L.EmbedID(n_vocab, n_units, initialW=I.Uniform(1. / n_units))

            if self.pos and self.wlen:
                assert(n_pos)
                n_inputs = (n_units + n_pos_units + 1) * (window + 1)
                self.embed_pos = L.EmbedID(n_pos, n_pos_units, initialW=I.Uniform(1. / n_pos_units))
            
            elif self.wlen:
                n_inputs = (n_units + 1) * (window + 1)
            
            elif self.pos:
                n_inputs = (n_units + n_pos_units) * (window + 1)
                self.embed_pos = L.EmbedID(n_pos, n_pos_units, initialW=I.Uniform(1. / n_pos_units))
            
            else:
                n_inputs = n_units * (window + 1)

            self.lin = L.Linear(n_inputs, n_hidden, initialW=I.Uniform(1. / n_inputs))

            self.layers = list()
            for i in range(n_layers - 1):
                self.layers.append(L.Linear(n_hidden, n_hidden, initialW=I.Uniform(1. / n_units)))
            self.layers.append(L.Linear(n_hidden, 1, initialW=I.Uniform(1. / n_units)))

            self.out = out
            self.loss_func = loss_func

    def __call__(self, inputs, target):
        i = (self._embed_input(inputs)) # called from superclass
        h = self.out(self.lin(i))
        for l in self.layers:
            h = self.out(l(h))

        loss = self.loss_func(h, target)
        reporter.report({'loss': loss}, self)
        return loss

    def inference(self, inputs):
        i = (self._embed_input(inputs)) # called from superclass
        h = self.out(self.lin(i))
        for l in self.layers:
            h = self.out(l(h))
        return h


def convert(batch, device):
    x, targets = batch
    n = len(x)
    inputs = list()
    for i in range(n):
        if device >= 0:
            x[i] = cuda.to_gpu(x[i])
        inputs.append(x[i])       
    if device >= 0:
        targets = cuda.to_gpu(targets)
    return tuple(inputs), targets

