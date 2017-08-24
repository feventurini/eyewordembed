## Training Neural Word Embeddings with Eye-tracking data
#### ABSTRACT
Several psycholinguistic studies have provided convincing evidence regarding the intrinsic relation of gaze measurements and word properties. In this work, we aim to explore the effect of eye-tracking data in learning word embeddings. We propose a multi-task approach in which we train word representations by jointly learning to predict gaze with a shared regression task. The results obtained are mixed: whilst the word embeddings' quality, as judged by means of word similarities and analogies, does not improve, we present evidence that the model yields \textit{specialized} word embeddings which are able to capture regularities related to reading times. We provide a critical analysis of the results and our simple approach, as well as suggesting possible directions for improvement.


#### The code:

> This code was developed in the context of my MSc dissertation at University of Edinburgh and initially was not supposed to be made public. For this reason, it does not carry all the documentation, details and maintainability that I would have provided in a full-fledged commercial project. Nevertheless, I decided to make this repository publicly available so that it might prove useful to somebody willing to tackle a similar project. 

In the code, you may often find referenced the "dataset" folder. It was used to contain all the datasets, the KenLM language models and all of the subsample datasets. The two corpora used were the Dundee Corpus<sup>[1](#myfootnote1)</sup>/Treebank<sup>[2](#myfootnote2)</sup> and the Gigaword<sup>[3](#myfootnote3)</sup> dataset 5th edition, available through the university-wide linense at UoE. The directory is not available in the public repository for two reasons: the corpora are both licensed, and the files are too big.

The models folder contains all the designed model and the supporting code. 
- all the file starting with "train_*" can be run to train the single models (e.g. multitask (with and without limited vocabulary), eyetracking standalone, word2vec standalone..): 
  - train_eyetracking.py takes self-explanatory command lines options, with a handy help interface. An example of use is:
  train_eyetracking.py -wl -pos -sur 
  (this runs the eyetracking task with word length, pos tag and surprisal as features)
  - train_multitask.py and train_multitask_limit_vocab.py are run with the configuration in the config files (see bottom of this list)
- the folders "test_*" were used for the various runs of the models. They currently contain the scripts used for running the tests, as well the final statistics and a couple relevant examples of trained models each (they also contain a generate_tsv.py file to generate the statistics from the models' logs)
- eyetracking.py contains the definition of the regression model, designed in chainer (also contains the classifier model, mentioned but not reported in the dissertation)
- the multitask setting is obtained through an extension class of chainer, included directy in train_multitask.py at the top (and in all of the scripts where multitask learning needs to happen)
- prepare_dataset.py is used to finish the preprocessing and prepares the eye-tracking data for training
- eyetracking_batch_iter.py is the data iterator for the datase of eyetracking measures obtained by prepare_dataset.py. It is designed to be flexible with regards to the choice of the features.
- multitask_batch_iter.py is the counterpart for iterating over the raw-text corpus, with some threading tricks for keeping the iteration synchronized with regards to the multitask. It was needed because gensim does not natively allow for batch training at sentence level, but in our case it is necessary for training alternatively on the two tasks and achieving multi-task learning.
- evaluate_eyetracking.py is for the evaluation of the models on the reading times prediction on the test set (it should be really be in the evaluation folder, but is here just for practical reasons)
- the "config_*" files contain the configuration for the finetuned and multitask models

The evaluation folder contains all the material used for evaluating the models
- the folder tensorboard_visualization contains supporting code for using the tensorboard word embeddings projector
- the folder eval-word-vectors contains readapted code from https://github.com/mfaruqui/eval-word-vectors, correctly MIT licensed
- evaluate.py is a script that I used for evaluating on GoogleAnalogyTestSet, WS-353, Simlex-999 and SimVerb-3500 (contained in evaluation_datasets). Gives more detailed results than the other, such as the subcategories for the analogy.

The preprocessing folder contains all the code use for prepreprocessing both the corpus:
- (note: as mentioned, part of the preprocessing for Dundee happens in models/prepare_dataset.py)
- downsample_dataset.py is the script used for downsampling the Gigaword dataset
- add_surprisal_dundee.py is the code used for adding surprisal to the dundee, given the KenLM n-gram models
- dundee_parser.py was used to parse and preprocess the Dundee Treebank and produced the two folders dataset/dundee_parsed_gr and dataset/dundee_parsed

The utilities folder contains supporting code (for I/O to file system and timing) used throughout the project

The bash_tools folder contains two scripts used for compressing and decompressing the datasets

The tsv_results contains the tsv file with the results appearing in the final dissertation tables

REMINDER: in case you need other supporting material (language models, trained models..), write me at venturini.fe@gmail.com or here on github

<a name="myfootnote1">1</a>: Kennedy, A., Hill, R., & Pynte, J. (2003, August). The dundee corpus. In Proceedings of the 12th European conference on eye movement. <br/>
<a name="myfootnote2">2</a>: Barrett, M., Agić, Ž., & Søgaard, A. (2015, January). The Dundee Treebank. In The 14th International Workshop on Treebanks and Linguistic Theories (TLT 14).<br/>
<a name="myfootnote3">3</a>: https://catalog.ldc.upenn.edu/ldc2011t07
 
