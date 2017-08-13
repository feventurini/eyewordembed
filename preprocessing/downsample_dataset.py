import os
import bz2
import tarfile
import random
import multiprocessing
import threading
import smart_open
import sys
sys.path.insert(0, '../utilities')
import timing


if __name__ == '__main__':
    tar = '../dataset/downsampled_gigaword/tokenized_gigaword_1024.tar.bz2'
    n = 3
    out_folder = 'downsampled_gigaword'
    if not os.path.isdir('downsampled_gigaword'):
        os.makedirs('downsampled_gigaword')


    maxsize = 8
    queue = multiprocessing.Queue(maxsize=maxsize*2)
    sentences = smart_open.smart_open(tar)
    batch_size = 100000

    def _batchFiller():
        counter = 0
        batch = list()
        for l in sentences:
            batch.append(l)
            counter += 1
            if not counter % 1000000:
                print('Processed {} lines'.format(counter))
            if len(batch) == batch_size:
                queue.put(batch, block=True)
                batch = list()
        queue.put(batch, block=True)
        for i in range(maxsize):
            queue.put(None)
  
    filler_thread = threading.Thread(target=_batchFiller) 
    filler_thread.daemon = True
    filler_thread.start()

    def worker(queue):
        pid = os.getpid()
        out_files = { 2**(i+1): open(os.path.join(out_folder, 'tokenized_gigaword_{}_{}'.format(pid, 2**(i+1))), 'w+') for i in range(n) }
    
        lines = queue.get(block=True)
        while lines:
            # for i in range(n):
            #     k = 2**(i+1)
            #     print(k, 1.0/k*len(lines), int(1.0/k*len(lines)))
            #     for l in random.sample(lines, int(1.0/k*len(lines))):
            #         out_files[k].write(l.decode('UTF-8'))
            for line in lines:
                for i in range(n):
                    k = 2**(i+1)
                    if random.random() <= 1.0/k:
                        out_files[k].write(line.decode('UTF-8'))
            lines = queue.get(True)

        for v in out_files.values():
            v.flush()
            v.close()

    pool = multiprocessing.Pool(maxsize, worker, (queue,))
    pool.close()
    pool.join()

    def compress(k):
        with tarfile.open('tokenized_gigaword_{}.tar.bz2'.format(k), 'w:bz2') as tar_out:
            for file in filter(lambda x: x.endswith('_{}'.format(k)), os.listdir('.')):
                tar_out.add(file)
        for file in filter(lambda x: x.endswith('_{}'.format(k)), os.listdir('.')):
            os.remove(file)

    os.chdir(out_folder)
    pool = multiprocessing.Pool(2)
    pool.map(compress, [2**(i+1) for i in range(n)])
    
    pool.close()
    pool.join()



    # with smart_open.smart_open(tar) as f:
    #     counter = 0
    #     for line in f:
    #         counter += 1
    #         if not counter % 1000000:
    #             print('Processed {} lines'.format(counter))
    #         if counter == 2000000:
    #             sys.exit(0)
    #         for i in range(n):
    #             k = 2**(i+1)
    #             if random.random() < 1.0/k:
    #                 out_files[k].write(line)

    # for v in out_files.values():
    #   v.close()

