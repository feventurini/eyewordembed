from collections import Counter
from itertools import chain
import os
from itertools import dropwhile
import timing

src_folder = "tokenized_gigaword"

def countInFile(filename):
    with open(filename) as f:
        return Counter(chain.from_iterable(map(str.split, f)))

def trimVocab(counter, frequency):
	for key, count in dropwhile(lambda key_count: key_count[1] >= frequency, main_dict.most_common()):
	    del main_dict[key]


c = 0
counter = Counter()
for filename in sorted(os.listdir(src_folder)):
	c+=1;
	print(filename)
	counter.update(countInFile(src_folder + '/' + filename))
	if c == 30:
		break

print(len(counter))
