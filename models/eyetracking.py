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

    def __call__(self, w, target):
        loss = self.loss_func(np.full(target.shape, self.mean, dtype=np.float32), target)
        reporter.report({'loss': loss}, self)
        return loss        

class LinReg(chainer.Chain):

    def __init__(self, n_vocab, n_units, loss_func, out, wlen=False, pos=False, n_pos=None, n_pos_units=50):
        super(LinReg, self).__init__()

        self.pos = pos
        self.wlen = wlen

        with self.init_scope():
            self.embed = L.EmbedID(n_vocab, n_units, initialW=I.Uniform(1. / n_units))

            if self.pos and self.wlen:
                assert(n_pos)
                self.embed_pos = L.EmbedID(n_pos, n_pos_units, initialW=I.Uniform(1. / n_pos_units))
                self.lin = L.Linear(n_units + n_pos_units + 1, 1, initialW=I.Uniform(1. / n_units))
            elif self.wlen:
                self.lin = L.Linear(n_units + 1, 1, initialW=I.Uniform(1. / n_units))
            elif self.pos:
                self.embed_pos = L.EmbedID(n_pos, n_pos_units, initialW=I.Uniform(1. / n_pos_units))
                self.lin = L.Linear(n_units + n_pos_units, 1, initialW=I.Uniform(1. / n_units))
            else:
                self.lin = L.Linear(n_units, 1, initialW=I.Uniform(1. / n_units))
            
            self.out = out
            self.loss_func = loss_func

    def _embed_input(self, inputs):
        if self.pos and self.wlen:
            w, l, p = inputs
            e_w = self.embed(w)
            e_p = self.embed_pos(p)
            l = F.reshape(l,(-1,1,1))
            h = F.concat((e_w, e_p, l), axis=2)
        elif self.pos:
            w, p = inputs
            e_w = self.embed(w)
            e_p = self.embed_pos(p)
            h = F.concat((e_w, e_p), axis=2)
        elif self.wlen:
            w, l = inputs
            e_w = self.embed(w)
            l = F.reshape(l,(-1,1,1))
            h = F.concat((e_w, l), axis=2)
        else:
            if isinstance(inputs, tuple):
                w = inputs[0]
            else:
                w = inputs
            h = self.embed(w)

        return h

    def __call__(self, inputs, target):
        h = self._embed_input(inputs)
        o = self.out(self.lin(h))
        loss = self.loss_func(o, target)
        reporter.report({'loss': loss}, self)
        return loss

    def inference(self, inputs):
        h = self._embed_input(inputs)
        return self.out(self.lin(h))

class Multilayer(LinReg):

    def __init__(self, n_vocab, n_units, loss_func, out, n_hidden=50, n_layers=1, wlen=False, pos=False, n_pos=None, n_pos_units=50):
        super(LinReg, self).__init__()

        self.pos = pos
        self.wlen = wlen

        with self.init_scope():
            self.embed = L.EmbedID(n_vocab, n_units, initialW=I.Uniform(1. / n_units))

            if self.pos and self.wlen:
                assert(n_pos)
                self.embed_pos = L.EmbedID(n_pos, n_pos_units, initialW=I.Uniform(1. / n_pos_units))
                self.lin = L.Linear(n_units + n_pos_units + 1, n_hidden, initialW=I.Uniform(1. / n_units))
            elif self.wlen:
                self.lin = L.Linear(n_units + 1, n_hidden, initialW=I.Uniform(1. / n_units))
            elif self.pos:
                self.embed_pos = L.EmbedID(n_pos, n_pos_units, initialW=I.Uniform(1. / n_pos_units))
                self.lin = L.Linear(n_units + n_pos_units, n_hidden, initialW=I.Uniform(1. / n_units))
            else:
                self.lin = L.Linear(n_units, n_hidden, initialW=I.Uniform(1. / n_units))
            
            self.layers = list()
            for i in range(n_layers - 1):
                self.layers.append(L.Linear(n_hidden, n_hidden, initialW=I.Uniform(1. / n_units)))
            self.layers.append(L.Linear(n_hidden, 1, initialW=I.Uniform(1. / n_units)))
            
            self.out = out
            self.loss_func = loss_func

    def _embed_input(self, inputs):
        if self.pos and self.wlen:
            w, l, p = inputs
            e_w = self.embed(w)
            e_p = self.embed_pos(p)
            l = F.reshape(l,(-1,1,1))
            h = F.concat((e_w, e_p, l), axis=2)
        elif self.pos:
            w, p = inputs
            e_w = self.embed(w)
            e_p = self.embed_pos(p)
            h = F.concat((e_w, e_p), axis=2)
        elif self.wlen:
            w, l = inputs
            e_w = self.embed(w)
            l = F.reshape(l,(-1,1,1))
            h = F.concat((e_w, l), axis=2)
        else:
            if inputs is tuple:
                w = inputs[0]
            else:
                w = inputs
            h = self.embed(w)

        return h

    def __call__(self, inputs, target):

        i = (self._embed_input(inputs))
        h = self.out(self.lin(i))
        for l in self.layers:
            h = self.out(l(h))

        loss = self.loss_func(h, target)
        reporter.report({'loss': loss}, self)
        return loss

class LinRegContextSum(chainer.Chain):

    def __init__(self, n_vocab, n_units, loss_func, out, window=2, wlen=False, pos=False, n_pos=None, n_pos_units=50):
        super(LinRegContextSum, self).__init__()

        self.pos = pos
        self.wlen = wlen

        with self.init_scope():
            self.embed = L.EmbedID(n_vocab, n_units, initialW=I.Uniform(1. / n_units))

            if self.pos and self.wlen:
                assert(n_pos)
                self.embed_pos = L.EmbedID(n_pos, n_pos_units, initialW=I.Uniform(1. / n_pos_units))
                self.lin = L.Linear(n_units + n_pos_units + 1, 1, initialW=I.Uniform(1. / n_units))
            elif self.wlen:
                self.lin = L.Linear(n_units + 1, 1, initialW=I.Uniform(1. / n_units))
            elif self.pos:
                self.embed_pos = L.EmbedID(n_pos, n_pos_units, initialW=I.Uniform(1. / n_pos_units))
                self.lin = L.Linear(n_units + n_pos_units, 1, initialW=I.Uniform(1. / n_units))
            else:
                self.lin = L.Linear(n_units, 1, initialW=I.Uniform(1. / n_units))

            self.out = out
            self.loss_func = loss_func

    def __call__(self, inputs, target):

        if self.pos and self.wlen:
            w, l, p = inputs
            e_w = F.sum(self.embed(w), axis=1) * (1. / w.shape[1])
            e_p = F.sum(self.embed_pos(p), axis=1) * (1. / p.shape[1])
            l = F.mean(l, axis=1).reshape(-1,1) #F.reshape(l,(-1,w.shape[1]))
            h = F.concat((e_w, e_p, l), axis=1)
        elif self.pos:
            w, p = inputs
            e_w = F.sum(self.embed(w), axis=1) * (1. / w.shape[1])
            e_p = F.sum(self.embed_pos(p), axis=1) * (1. / p.shape[1])
            h = F.concat((e_w, e_p), axis=1)
        elif self.wlen:
            w, l = inputs
            e_w = F.sum(self.embed(w), axis=1) * (1. / w.shape[1])
            l = F.mean(l, axis=1).reshape(-1,1) #F.reshape(l,(-1,w.shape[1]))
            h = F.concat((e_w, l), axis=1)
        else:
            w = inputs
            h = F.sum(self.embed(w), axis=1) * (1. / w.shape[1])

        o = self.out(self.lin(h))
        loss = self.loss_func(o, target)
        reporter.report({'loss': loss}, self)
        return loss

class LinRegContextConcat(chainer.Chain):

    def __init__(self, n_vocab, n_units, loss_func, out, window=2, wlen=False, pos=False, n_pos=None, n_pos_units=50):
        super(LinRegContextConcat, self).__init__()

        self.n_units = n_units
        self.n_pos_units = n_pos_units
        self.pos = pos
        self.wlen = wlen

        with self.init_scope():
            self.embed = L.EmbedID(n_vocab, n_units, initialW=I.Uniform(1. / n_units))

            if self.pos and self.wlen:
                assert(n_pos)
                self.embed_pos = L.EmbedID(n_pos, n_pos_units, initialW=I.Uniform(1. / n_pos_units))
                self.lin = L.Linear((n_units + n_pos_units + 1)*window, 1, initialW=I.Uniform(1. / n_units))
            elif self.wlen:
                self.lin = L.Linear((n_units + 1)*window, 1, initialW=I.Uniform(1. / n_units))
            elif self.pos:
                self.embed_pos = L.EmbedID(n_pos, n_pos_units, initialW=I.Uniform(1. / n_pos_units))
                self.lin = L.Linear((n_units + n_pos_units)*window, 1, initialW=I.Uniform(1. / n_units))
            else:
                self.lin = L.Linear((n_units)*window, 1, initialW=I.Uniform(1. / n_units))

            self.out = out
            self.loss_func = loss_func

    def __call__(self, inputs, target):

        if self.pos and self.wlen:
            w, l, p = inputs
            e_w = F.reshape(self.embed(w), (-1,w.shape[1]*self.n_units))
            e_p = F.reshape(self.embed_pos(p), (-1,w.shape[1]*self.n_pos_units))
            h = F.concat((e_w, e_p, l), axis=1)# * (1. / w.shape[1])
        elif self.pos:
            w, p = inputs
            e_w = F.reshape(self.embed(w), (-1,w.shape[1]*self.n_units))
            e_p = F.reshape(self.embed_pos(p), (-1,w.shape[1]*self.n_pos_units))
            h = F.concat((e_w, e_p), axis=1)
        elif self.wlen:
            w, l = inputs
            e_w = F.reshape(self.embed(w), (-1,w.shape[1]*self.n_units))
            h = F.concat((e_w, l), axis=1)
        else:
            if inputs is tuple:
                w = inputs[0]
            else:
                w = inputs
            h = F.reshape(self.embed(w), (-1,w.shape[1]*self.n_units))

        o = self.out(self.lin(h))
        loss = self.loss_func(o, target)
        reporter.report({'loss': loss}, self)
        return loss

    def inference(self, inputs):
        if self.pos and self.wlen:
            w, l, p = inputs
            e_w = F.reshape(self.embed(w), (-1,w.shape[1]*self.n_units))
            e_p = F.reshape(self.embed_pos(p), (-1,w.shape[1]*self.n_pos_units))
            h = F.concat((e_w, e_p, l), axis=1)# * (1. / w.shape[1])
        elif self.pos:
            w, p = inputs
            e_w = F.reshape(self.embed(w), (-1,w.shape[1]*self.n_units))
            e_p = F.reshape(self.embed_pos(p), (-1,w.shape[1]*self.n_pos_units))
            h = F.concat((e_w, e_p), axis=1)
        elif self.wlen:
            w, l = inputs
            e_w = F.reshape(self.embed(w), (-1,w.shape[1]*self.n_units))
            h = F.concat((e_w, l), axis=1)
        else:
            w = inputs[0]
            h = F.reshape(self.embed(w), (-1,w.shape[1]*self.n_units))

        return self.out(self.lin(h))


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

