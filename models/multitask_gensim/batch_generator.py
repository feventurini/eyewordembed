import os
import random
import itertools
import numpy as np

class BatchIterator(object):
	def __init__(self, sentences, batch_size=1000):
		self.sentences = sentences
		self.batch_size = batch_size
		self.index = 0

	def next(self):
		end = self.index + self.batch_size
		batch = itertools.islice(self.sentences, self.index, end)
		self.index += len(batch)
		if self.index >= len(sentences):
			self.index = 0
		return batch