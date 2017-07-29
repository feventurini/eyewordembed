import chainer
import numpy as np

class EyetrackingBatchIterator(chainer.dataset.Iterator):

    def __init__(self, dataset, window, batch_size, repeat=True, shuffle=True, 
        wlen=False, pos=False, prev_fix=False, freq=False, surprisal=False, bins=False):
        self.words = dataset[:,0].astype(np.int32)#.reshape(-1,1)
        self.bins = bins
        if bins:
            self.times = dataset[:,1].astype(np.int32)
            self.previous_times = dataset[:,4].astype(np.int32)
        else:
            self.times = dataset[:,1].astype(np.float32)
            self.previous_times = dataset[:,4].astype(np.float32)
        self.wlens = dataset[:,2].astype(np.float32)
        self.pos_tags = dataset[:,3].astype(np.int32)
        self.freqs = dataset[:,5].astype(np.float32)
        self.surprisals = dataset[:,6].astype(np.float32)

        self.wlen = wlen
        self.pos = pos
        self.prev_fix = prev_fix
        self.freq = freq
        self.surprisal = surprisal

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
        if self.bins:
            times = self.times.take(position).reshape(-1,)
        else:
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

        out = {}
        out['words'] = context
        if self.prev_fix:
            out['prev_fix'] = self.previous_times.take(pos)
        if self.pos:
            out['pos'] = self.pos_tags.take(pos)
        if self.wlen:
            out['wlen'] = self.wlens.take(pos)
        if self.freq:
            out['freq'] = self.freqs.take(pos)
        if self.surprisal:
            out['surprisal'] = self.surprisals.take(pos)

        return out, times

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