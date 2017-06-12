import os
import tarfile
import numpy as np

def listfiles(dir):
    l = list()
    for r,_,f in os.walk(dir):
        l += map(lambda filename: r + '/' + filename, f)
    return l

def readcompressed(file):
    with tarfile.open(file) as tar:
        with tar.extractfile(tar.next()) as f:
            return list(map(lambda x : x.strip().decode('UTF-8'), f))
            #return list(map(lambda x : np.array(x.strip().split()), f))
 
def readcompressed2(file):
    with tarfile.open(file) as tar:
        with tar.extractfile(tar.next()) as f:
            l = list()
            for line in f:
                l += line.strip().split()
        return np.array(l, dtype=np.int32)

print(readcompressed2(listfiles('compressed_gigaword_dataset')[2]))