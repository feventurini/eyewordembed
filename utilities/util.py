import pickle
import gzip

def save(obj, path):
    with gzip.open(path + '.pklz','wb') as f:
        pickle.dump(obj, f)  

def load(path):
    with gzip.open(path + '.pklz','rb') as f:
        return pickle.load(f)    
