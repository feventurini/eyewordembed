import os
import string
import re
import parallel_vocab_creator as pvc
import timing
from multiprocessing import Pool
import logging
# import numpy as np
# import pickle
# import gzip

global word2id
word2id_path = 'word2id'
word2id = pvc.load(word2id_path)

global unk_token 
unk_token = word2id['<UNK>']

#id2word_path = 'id2word'
#global id2word
#id2word = pvc.load(id2word_path)


def convert_op(filename):
    logger.info("Currently converting file: " + filename)
    mapper = Token2IdMapper(src_folder, filename, dest_folder)
    mapper.convert()
    mapper.close()

class Token2IdMapper():
    
    def __init__(self, src_folder, filename, dest_folder):
        self.out_path = dest_folder + '/' + filename
        self.out_file = open(dest_folder + '/' + filename, 'w')
        self.in_file = open(src_folder + '/' + filename, 'r')

    def get_from_dict(self, w):
        try:
            return word2id[w]
        except:
            return unk_token
    
    def convert_line(self, line):       
        ids = list(map(self.get_from_dict, line.split()))
        print(' '.join(map(str,ids)), file=self.out_file)

        #return np.array(list(map(self.get_from_dict, line.split())))

    def convert(self):
        res = list()
        for line in self.in_file:
            self.convert_line(line)

            #res.append(self.convert_line(line))
        #pvc.save(res, self.out_path)

    def close(self):
        self.in_file.close()
        self.out_file.close()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    src_folder = 'tokenized_gigaword'
    dest_folder = 'gigaword_dataset'
    logger.info("Converting corpus in " + src_folder + " to token ids, results in folder " + dest_folder + "...")
    logger.info("Starting pool of 4 processes...")
    
    pool = Pool(4)
    pool.map(convert_op, os.listdir(src_folder))

    logger.info("Task finished")
