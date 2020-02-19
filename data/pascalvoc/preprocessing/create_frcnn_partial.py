import sys

import numpy as np

from config.config import cfg

from collections import defaultdict
from pprint import pprint


def create_partial(proportion, subset):
    '''
    proportion = percentage of the original dataset to keep = probability of keeping the line

    TODO: make sure the random seed cannot be overwritten when launching the model (currently it is not, but not enforced)
    '''
    print('computing partial with p %s' % proportion)

    p = float(proportion) / 100  # original is out of 100, prob is [0, 1]

    np.random.seed(cfg.RANDOM_SEED)

    original_path = '/local/DEEPLEARNING/pascalvoc/VOCdevkit/VOC2007/Annotations/frcnn_%s.csv' % subset
    partial_path = '/local/DEEPLEARNING/pascalvoc/VOCdevkit/VOC2007/Annotations/frcnn_%s_partial_%s_%s.csv' % (subset, proportion, cfg.RANDOM_SEED)

    count = 0
    count_new = 0

    count_per_class = defaultdict(lambda: 0)
    count_new_per_class = defaultdict(lambda: 0)

    with open(original_path, 'r') as f_in, open(partial_path, 'w+') as f_out:
        for line in f_in:
            count += 1
            keep = np.random.random() < p

            parts = line.split(',')
            count_per_class[parts[1]] += 1

            if keep:
                count_new += 1
                f_out.write(line)
                count_new_per_class[parts[1]] += 1

    print('for subset %s, original nb of lines = %s, new nb of lines = %s' % (subset, count, count_new))

    print('count per class')
    pprint(count_per_class)

    print('count new per class')
    pprint(count_new_per_class)


# python3 create_frcnn_partial.py 10 trainval_ext
if __name__ == '__main__':
    subset = sys.argv[2]
    proportion = sys.argv[1]
    create_partial(proportion, subset)
