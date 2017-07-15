import gensim
import logging
import argparse
import os
import sys

## EVALUATION DATASETS HARD CODED:
analogy_questions_file_google = './evaluation_datasets/analogy_google_questions.txt'
ws_353_dataset = './evaluation_datasets/wordsim353.txt'
simlex_999_dataset = './evaluation_datasets/SimLex-999.txt'
bats_folder = './evaluation_datasets/BATS_3.0'

def analogy_accuracy(model, questions_file):
    acc = model.accuracy(questions_file)

    correct = sum([len(acc[i]['correct']) for i in range(5)])
    total = sum([len(acc[i]['correct']) + len(acc[i]['incorrect']) for i in range(5)])
    accuracy = 100*float(correct)/total
    logging.info('\nSemantic: {:d}/{:d}, Accuracy: {:.2f}%'.format(correct, total, accuracy))
    
    correct = sum((len(acc[i]['correct']) for i in range(5, len(acc)-1)))
    total = sum((len(acc[i]['correct']) + len(acc[i]['incorrect'])) for i in range(5,len(acc)-1))
    accuracy = 100*float(correct)/total
    logging.info('Syntactic: {:d}/{:d}, Accuracy: {:.2f}%\n'.format(correct, total, accuracy))
    
def similarity_accuracy(model, similarity_file):
    acc = model.evaluate_word_pairs(similarity_file)
    logging.info('Pearson correlation coefficient: {:.2f}'.format(acc[0][0]))
    logging.info('Spearman rank correlation coefficient: {:.2f}'.format(acc[1][0]))

def bats_analogy(model, bats_folder, output_folder):
    if not os.path.isdir('vsmlib'):
        logging.error('Can''t find vsmlib folder for BATS analogy evaluation')
        return
    temp_config_file = os.path.join('vsmlib', 'scripts', 'temp_config.yaml')
    vectors_file = os.path.join('vsmlib', 'scripts', 'vectors.bin')

    with open(temp_config_file, 'w+') as config:
        config.write('type_vectors:\n') 
        config.write('path_vectors: [.]\n')
        config.write('method: 3CosAdd\n')
        config.write('exclude: True\n')
        config.write('path_dataset: ' + os.path.abspath(bats_folder) + '\n')
        config.write('path_results: ' + os.path.abspath(output_folder) + '\n')
    model.save_word2vec_format(vectors_file, binary=True)

    old_dir = os.getcwd()
    os.chdir('./vsmlib/scripts')
    os.system('python test_analogy.py temp_config.yaml')
    os.chdir(old_dir)
    os.remove(temp_config_file)
    os.remove(vectors_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument( 'input', metavar='input', type=str, help="Input word2vec model")
    parser.add_argument( '-o', '--output', default='eval_log', help="Log output folder")
    parser.add_argument( "-g", "--gensim", required=False, action='store_true', help="Add this option if the model is in gensim format")
    parser.add_argument( "-b", "--binary", required=False, action='store_true', help="If word2vec model in binary format, set True, else False")
    args = parser.parse_args()

    if not os.path.isdir(args.output):
        os.makedirs(args.output)

    logging.basicConfig(filename=os.path.join(args.output,args.input.split(os.sep)[-1] + '.log'), filemode='w', format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    # load model
    if args.gensim:
        model = gensim.models.Word2Vec.load(args.input).wv
    else:
        model = gensim.models.KeyedVectors.load_word2vec_format(args.input, binary=args.binary)

    analogy_accuracy(model, analogy_questions_file_google)
    similarity_accuracy(model, ws_353_dataset)
    similarity_accuracy(model, simlex_999_dataset)
    # bats_analogy(model, bats_folder, args.output)
