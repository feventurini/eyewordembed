import os
import random
import chainer
import tarfile
import numpy as np

def listfiles(dir):
    l = list()
    for r,_,f in os.walk(dir):
        l += map(lambda filename: r + '/' + filename, f)
    return l

def readcompressed2(file):
    with tarfile.open(file) as tar:
        with tar.extractfile(tar.next()) as f:
            return list(map(lambda x : x.strip().decode('UTF-8'), f))
            #return list(map(lambda x : np.array(x.strip().split()), f))
 
def readcompressed(tarinfo, tarball):
        f = tarball.extractfile(tarinfo)
        l = list()
        for line in f:
            l += line.strip().split()
        return np.array(l, dtype=np.int32)

class GigawordBatchIterator(chainer.dataset.Iterator):

    def __init__(self, tar_file, window, batch_size, repeat=True):
        self.tarball = tarfile.open(tar_file)
        self.files = self.tarball.getmembers()
        random.shuffle(self.files)
        self.current_file_index = -1

        self.window = window

        self.load_next_file()

        self.batch_size = batch_size
        self._repeat = repeat


        self.current_position = 0
        self.epoch = 0
        self.is_new_epoch = False

        self.order += window

    def load_next_file(self):
        self.current_file_index += 1
        self.current_file = readcompressed(self.files[self.current_file_index], self.tarball)
        self.current_position = 0
        self.order = np.random.permutation(len(self.current_file) - self.window * 2).astype(np.int32)

    def __next__(self):
        if not self._repeat and self.epoch > 0:
            raise StopIteration

        i = self.current_position
        i_end = i + self.batch_size
        position = self.order[i: i_end]
        w = np.random.randint(self.window - 1) + 1
        offset = np.concatenate([np.arange(-w, 0), np.arange(1, w + 1)])
        pos = position[:, None] + offset[None, :]
        context = self.current_file.take(pos)
        center = self.current_file.take(position)

        if i_end >= len(self.order):
            print("FINISHED FILE, GOING TO NEXT ONE")
            if self.current_file_index < len(self.files) - 1:
                self.load_next_file()
                self.is_new_epoch = False
            else:
                self.is_new_epoch = True
                self.current_position = 0
                self.epoch += 1     
                random.shuffle(self.files)
                self.current_file_index = -1
                self.load_next_file()
        else:
            self.is_new_epoch = False
            self.current_position = i_end

        print (center, context)
        return center, context

    @property
    def epoch_detail(self):
        return self.epoch + float(self.current_file_index) / len(self.files) + float(self.current_position) / len(self.order)

    def serialize(self, serializer):
        self.current_position = serializer('current_position', self.current_position)
        self.current_file_index = serializer('current_file_index', self.current_file_index)
        self.epoch = serializer('epoch', self.epoch)
        self.is_new_epoch = serializer('is_new_epoch', self.is_new_epoch)
        if self._order is not None:
            serializer('_order', self._order)
