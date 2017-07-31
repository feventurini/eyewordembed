import sys
sys.path.insert(0, '../utilities')
sys.path.insert(0, '../preprocessing')
import gensim
import util
import timing
import numpy as np
import scipy
import nltk
import argparse
import dundee_parser as dp
import csv
import string

def string_lower_nopunct_cast(s):
    return s.strip(string.punctuation).lower()

def float_or_blank_cast(s):
    try:
        return float(s)
    except:
        return 0.0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument( 'input', metavar='input', type=str, help="Input word2vec model")
    parser.add_argument( '-o', '--output', default='eval_log', help="Log output folder")
    parser.add_argument( "-g", "--gensim", required=False, action='store_true', help="Add this option if the model is in gensim format")
    parser.add_argument( "-b", "--binary", required=False, action='store_true', help="If word2vec model in binary format, set True, else False")
    args = parser.parse_args()

    if args.gensim:
        model = gensim.models.Word2Vec.load(args.input).wv
    else:
        model = gensim.models.KeyedVectors.load_word2vec_format(args.input, binary=args.binary)

    csv_path = '/media/fede/fedeProSD/eyewordembed/dataset/dundee_eyemovement/treebank/en_Dundee_DLT_freq_goldtok_gr.csv'

    dtp = dp.DundeeTreebankParser()

    with open(csv_path,'r') as f:
        r = csv.reader(f, delimiter='\t')
        firstrow = next(r)
        dtp.initiateParser(firstrow)
        sentence = []
        reading_times = []
        excluded = 0

        sentence_num = -1
        semdists = []
        times = []
        for row in r:
            if sentence_num == -1:
                sentence_num = row[dtp.map['SentenceID']]
            if sentence_num != row[dtp.map['SentenceID']]:
                sentence_num = row[dtp.map['SentenceID']]

                for i in range(1,len(sentence)):
                    context = np.array([model[sentence[j]] for j in range(max(0,i-4),i)]) # range(0, i) 
                    context = np.sum(context, axis=0)
                    semdists.append(scipy.spatial.distance.cosine(model[sentence[i]],context))
                    times.append(reading_times[i])
                sentence = []
                reading_times = []

            temp = list(filter(lambda x: x!='', map(string_lower_nopunct_cast, nltk.word_tokenize(row[dtp.map['Word']]))))
            sub_words = list(filter(lambda x: x in model, temp))
            excluded += (len(temp) - len(sub_words))
            sentence += sub_words
            t = float_or_blank_cast(row[dtp.map['Tot_fix_dur']])
            tt = [t for i in range(len(sub_words))]
            reading_times += tt

    # print(times)
    # print(semdists)

    # mask = np.array(times) != 0.0
    # print(mask)
    # semdists = np.array(semdists)[mask]
    # times = np.array(times)[mask]
    # input(times)

    print('Excluded tokens: {} out of {}'.format(excluded, len(semdists)))
    print('Pearson correlation coefficient: {}'.format(scipy.stats.pearsonr(semdists, times)[0]))