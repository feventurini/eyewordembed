import os
import random
import itertools
import numpy as np
from random import shuffle
import chainer
from queue import Queue
import threading
from gensim import utils

class BatchIterator(object):

    def __init__(self, sentences, epochs, total_examples, batch_size=100000):
        self.sentences = sentences
        self.batch_size = batch_size
        self.index = 0
        self.current_epoch = 0
        self.total_examples = total_examples
        self.queue = Queue(maxsize=5)
        self.closed_queue = False

        if epochs > 1:      
            sentences = utils.RepeatCorpusNTimes(sentences, epochs)

        def _batchFiller():
            counter = 0
            batch = list()
            for l in sentences:
                batch.append(l)
                counter += 1
                if len(batch) == batch_size:
                    self.queue.put(batch, block=True)
                    batch = list()
                if counter > total_examples:
                    self.current_epoch += 1
                    counter = 0
            self.queue.put(None)
            self.closed_queue = True

        self.filler_thread = threading.Thread(target=_batchFiller) 
        self.filler_thread.daemon = True
        self.filler_thread.start()

    def next(self):
        return None if self.closed_queue else self.queue.get(block=True, timeout=20)

class EyeTrackingSerialIterator(chainer.iterators.SerialIterator):

    def __next__(self):
        batch = super(EyeTrackingSerialIterator, self).__next__()
        x = np.array([b[0] for b in batch], dtype=np.int32).reshape(-1,1)
        targets = np.array([b[1] for b in batch], dtype=np.float32).reshape(-1,1)
        return x, targets

    next = __next__

class EyeTrackingWindowIterator(chainer.dataset.Iterator):

    def __init__(self, dataset, window, batch_size, repeat=True, shuffle=True):
        self.words = np.array([b[0] for b in dataset], dtype=np.int32).reshape(-1,1)
        self.times = np.array([b[1] for b in dataset], dtype=np.float32).reshape(-1,1)
        
        self.window = window
        self.batch_size = batch_size
        self._repeat = repeat
        self._shuffle = shuffle

        if self._shuffle:
            self.order = np.random.permutation(len(dataset) - window * 2).astype(np.int32)
        else:
            self.order = np.range(len(dataset) - window * 2).astype(np.int32)

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
        w = np.random.randint(self.window - 1) + 1
        offset = np.arange(-w, w)
        pos = position[:, None] + offset[None, :]
        context = self.words.take(pos)
        target = self.times.take(position).reshape(-1,1)

        if i_end >= len(self.order):
            if self._shuffle:
                np.random.shuffle(self.order)
            self.epoch += 1
            self.is_new_epoch = True
            self.current_position = 0
        else:
            self.is_new_epoch = False
            self.current_position = i_end

        return context, target

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

    