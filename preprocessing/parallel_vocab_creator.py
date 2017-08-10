from collections import Counter
from itertools import chain, dropwhile
import multiprocessing
import os
import pickle
import gzip
import logging
import gensim
from queue import Queue
import threading
    
    
import sys
sys.path.insert(0, '../utilities')
import timing
import util

window = 5
def log_result(result):
    l.acquire()
    vocab_context.update(result)
    l.release()

def countInBatch(batch, vocab):
    result = []
    for line in batch:
        for i in range(len(line)):
            if line[i] in result:
                continue
            context = line[max(0,i-window): min(len(line),i+window+1)]
            if any([d in context for d in vocab]):
                result.append(line[i])
    logging.info('Batch done...')
    return result

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
    vocab = set()
    with open('/media/fede/fedeProSD/eyewordembed/dataset/dundee_vocab.txt') as f:
        for line in f:
            vocab.update(line.split())

    src_file = "/media/fede/fedeProSD/eyewordembed/dataset/downsampled_gigaword/tokenized_gigaword_1024.tar.bz2"
    sentences = gensim.models.word2vec.LineSentence(src_file)

    queue = Queue(maxsize=5)
    closed_queue = False
    batch_size = 10000

    def _batchFiller():
        counter = 0
        batch = list()
        for l in sentences:
            batch.append(l)
            counter += 1
            if len(batch) == batch_size:
                queue.put(batch, block=True)
                batch = list()
        queue.put(batch, block=True)
        for i in range(4):
            queue.put(None)
        closed_queue = True

    filler_thread = threading.Thread(target=_batchFiller) 
    filler_thread.daemon = True
    filler_thread.start()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    if not os.path.isdir('../vocab'):
        os.makedirs('../vocab')

    os.chdir('../vocab')

    if not os.path.isfile('word_count.pklz'):
        pool = multiprocessing.Pool(4)
        vocab_context = Counter()
        logger.info("Creating vocab with context from file: '" + src_file + "'")
        l = multiprocessing.Lock()
        batch = queue.get()
        while not closed_queue and batch:
            pool.apply_async(countInBatch, args=(batch,vocab,), callback = log_result)
            # countInBatch(batch, vocab)
            batch = queue.get()

        pool.close()
        pool.join()

        input(vocab_context)
        logger.info("Saving word count to file 'word_count.pklz'")
        save(vocab_context, 'word_count')

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
