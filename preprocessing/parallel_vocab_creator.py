from collections import Counter
from itertools import chain, dropwhile
import multiprocessing
import os
import pickle
import gzip
import logging
	
import sys
sys.path.insert(0, '../utilities')
import timing
import util

def log_result(result):
    l.acquire()
    counter.update(result)
    l.release()

def countInFile(filename):
    logger.debug(filename)
    with open(filename) as f:
        return Counter(chain.from_iterable(map(str.split, f)))

def trimVocab(counter, frequency, max_size):
    unk_count = 0
    for key, count in dropwhile(lambda key_count: key_count[1] >= frequency, counter.most_common()):
        unk_count += count
        del counter[key]
    counter['<UNK>'] = unk_count
    # vocab = counter.most_common(max_size - 1)
    # vocab.append(('<UNK>',unk_count))
    return dict(counter.most_common(max_size))

def save(obj, path):
    with gzip.open(path + '.pklz','wb') as f:
        pickle.dump(obj, f)  

def load(path):
    with gzip.open(path + '.pklz','rb') as f:
        return pickle.load(f)    


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    src_folder = "/media/fede/fedeProSD/eyewordembed/dataset/training_decomp"

    os.chdir('../vocab')
    if not os.path.isfile('word_count.pklz'):
        pool = multiprocessing.Pool(4)
        counter = Counter()
        logger.info("Counting words from src folder: '" + src_folder + "'")
        l = multiprocessing.Lock()
        for filename in sorted(os.listdir(src_folder)):
            pool.apply_async(countInFile, args=(src_folder + '/' + filename,), callback = log_result)

        pool.close()
        pool.join()

        logger.info("Saving word count to file 'word_count.pklz'")
        save(counter, 'word_count')

    logger.info("Loading word_count from disk...")
    word_count = load('word_count')

    frequency = 5
    logger.info("Trimming vocabulary, excluding words with frequency less than " + str(frequency) + "...")
    old = len(word_count)
    vocab = trimVocab(word_count, frequency, 400000)

    logger.info("Saving trimmed word count to file 'word_count_trimmed.pklz'")
    save(vocab, 'word_count_trimmed')

    new = len(vocab)
    logger.info(str(old-new) + " types turned into <UNK>")
    logger.info("Vocabulary size: " + str(new))

    
    logger.info("Generating word2id...")
    word2id = {word: i for i, (word, count) in enumerate(vocab.items())}

    logger.info("Saving word2id...")
    save(word2id, 'word2id')

    logger.info("Generating id2word...")
    id2word = {v: k for k, v in word2id.items()}
    
    logger.info("Saving id2word...")
    save(id2word, 'id2word')
