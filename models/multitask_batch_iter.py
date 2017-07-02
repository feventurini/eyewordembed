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