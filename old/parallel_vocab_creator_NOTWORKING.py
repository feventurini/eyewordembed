import collections
from itertools import chain, dropwhile
import multiprocessing
import os
import timing
import pickle
import gzip
# import sys

global l
l = multiprocessing.Lock()

class ParallelWordCount():
    def __init__(self, src_folder, num_processes):
        self.src_folder = src_folder
        self.counter = collections.Counter()
        self.num_processes = num_processes

    def log_result(self, result):
        l.acquire()
        print(len(counter))
        self.counter.update(result)
        l.release()

    def countInFile(self, filename):
        with open(self.src_folder + '/' + filename) as f:
            print(filename)
            return Counter(chain.from_iterable(map(str.split, f)))

    def trimVocab(self, counter, frequency):
        for key, count in dropwhile(lambda key_count: key_count[1] >= frequency, main_dict.most_common()):
            del main_dict[key]

    def save(self, path):
        with gzip.open(path + '.pklz','wb') as f:
            pickle.dump(self.counter, f)

    def count(self):
        pool = multiprocessing.Pool(self.num_processes)

        for filename in sorted(os.listdir(self.src_folder)):
            pool.apply_async(self.countInFile, args=(filename,), callback = self.log_result)

        pool.close()
        pool.join()

if __name__ == '__main__':
    src_folder = "tokenized_gigaword"
    wc = ParallelWordCount(src_folder, 4)
    wc.count()
    wc.save('word_count')