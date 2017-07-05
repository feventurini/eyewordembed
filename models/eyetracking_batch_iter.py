import chainer
import numpy as np

class EyeTrackingWindowIterator(chainer.dataset.Iterator):

    def __init__(self, dataset, window, batch_size, repeat=True, shuffle=True, lens=False, pos=False):
        self.words = dataset[:,0].astype(np.int32)#.reshape(-1,1)
        self.times = dataset[:,1].astype(np.float32)
        self.wlens = dataset[:,2].astype(np.float32)
        self.pos_tags = dataset[:,3].astype(np.int32)
        
        self.lens = lens
        self.pos = pos

        self.window = window
        self.batch_size = batch_size
        self._repeat = repeat
        self._shuffle = shuffle

        if self._shuffle:
            self.order = np.random.permutation(len(dataset) - window * 2).astype(np.int32)
        else:
            self.order = np.arange(len(dataset) - window * 2).astype(np.int32)

        self.order += window
        self.current_position = 0
        self.epoch = 0
        self.is_new_epoch = False

    def __next__(self):
        if not self._repeat and self.epoch > 0:
            raise StopIteration

        i = self.current_position
        i_end = i + self.batch_size
        position = self.order[i: i_end]
        #w = np.random.randint(self.window - 1) + 1
        offset = np.arange(-self.window, 1)
        pos = position[:, None] + offset[None, :]


        context = self.words.take(pos)
        times = self.times.take(position).reshape(-1,1)

        if i_end >= len(self.order):
            if self._shuffle:
                np.random.shuffle(self.order)
            self.epoch += 1
            self.is_new_epoch = True
            self.current_position = 0
        else:
            self.is_new_epoch = False
            self.current_position = i_end


        if self.lens and self.pos:
            ls = self.wlens.take(pos)
            ps = self.pos_tags.take(pos)
            return (context, ls, ps), times
        elif self.lens:
            ls = self.wlens.take(pos)
            return (context, ls), times 
        elif self.pos:
            ps = self.pos_tags.take(pos)
            return (context, ps), times
        else:
            return (context,), times

    @property
    def epoch_detail(self):
        return self.epoch + float(self.current_position) / len(self.order)

    def serialize(self, serializer):
        self.current_position = serializer('current_position',
                                           self.current_position)
        self.epoch = serializer('epoch', self.epoch)
        self.is_new_epoch = serializer('is_new_epoch', self.is_new_epoch)
        if self._order is not None:
            serializer('_order', self._order)

class EyeTrackingSerialIterator(chainer.iterators.SerialIterator):
    def __init__(self, *args, **kwargs):
        self.lens = kwargs.pop('lens', False)
        self.pos = kwargs.pop('pos', False)
        super().__init__(*args, **kwargs)

    def __next__(self):
        batch = super().__next__()
        words, times, ls, ps = map(np.array, zip(*[tuple(l) for l in batch]))
        words = words.astype(np.int32).reshape(-1,1)
        times = times.astype(np.float32).reshape(-1,1)

        if self.lens and self.pos:
            ls = ls.astype(np.float32).reshape(-1,1)
            ps = ps.astype(np.int32).reshape(-1,1)
            return (words, ls, ps), times
        elif self.lens:
            ls = ls.astype(np.float32).reshape(-1,1)
            return (words, ls), times 
        elif self.pos:
            ps = ps.astype(np.int32).reshape(-1,1)
            return (words, ps), times
        else:
            return (words,), times
    
    next = __next__    



