import os
import sys

import numpy as np
from collections import defaultdict


def create_positional_prior(dataset_path):
    '''
    prior matrix size: (24, 24, 4)
    24 x 24 for all classes with all classes
    4 for each prior type:
        - 0: inside
        - 1: intersect
        - 2: above
        - 3: below

    example line:
    000012,7,156,97,351,270,500,333,3
    '''

    count_matrix = np.zeros((24, 24)) * 0.0001           # absolute counts of pairs happening together
    prior_count_matrix = np.zeros((24, 24, 4))  # absolute counts of positional relations
    prior_matrix = np.zeros((24, 24, 4))        # storing in percent
    data = defaultdict(list)

    # get relevant info from dataset
    with open(dataset_path, 'r') as f_in:
        for line in f_in:
            parts = line.strip().split(',')

            data[parts[0]].append(parts[1:6])   # just get the relevant parts

    # for each image we examine each pair of objects
    for image_id, image_data in data.items():

        # double loop to examine all pairs
        for i, obj_i in enumerate(image_data):
            for j, obj_j in enumerate(image_data):

                # don't count pairs with ourselves
                if i == j:
                    continue

                class_i = int(obj_i[0])
                class_j = int(obj_j[0])

                # increment counting
                prior_matrix[class_i][class_j] += 1
                prior_matrix[class_j][class_i] += 1

                # check any relationship
                relation_ij = check_relationship(obj_i, obj_j)
                relation_ji = check_relationship(obj_j, obj_i)

                if relation_ij is not None:
                    prior_count_matrix[class_i][class_j] = relation_ij

                if relation_ji is not None:
                    prior_count_matrix[class_j][class_i] = relation_ji

    # compute the prior matrix in percent
    expanded_count = np.repeat(count_matrix[:, :, np.newaxis], 4, axis=2)
    prior_matrix = prior_count_matrix * 100 / expanded_count

    # saving as files
    dataset_folder = os.path.dirname(dataset_path)
    np.save(os.path.join(dataset_folder, 'prior_counting'), count_matrix)
    np.save(os.path.join(dataset_folder, 'prior_relations_count'), prior_count_matrix)
    np.save(os.path.join(dataset_folder, 'prior_matrix'), prior_matrix)


def check_relationship(object_1, object_2):
    '''
    Compute 4 types of relationships
    Returns None if no relationship is found
    Relations:
    - 0: object_1 is    __inside__ object_2
    - 1: object_1       __intersects__ with object_2
    - 2: object_1 is    __above__ object_2
    - 3: object_1 is    __below__ object_2
    '''

    xmin1, ymin1, xmax1, ymax1 = object_1[1:]
    xmin2, ymin2, xmax2, ymax2 = object_2[1:]

    # inside
    if (xmin1 > xmin2) and (xmax1 < xmax2) and (ymin1 > ymin2) and (ymax1 < ymax2):
        return 0

    # intersect
    if ((ymin1 < ymin2 < ymax1) or (ymin2 < ymin1 < ymax2)) and ((xmin1 < xmin2 < xmax1) or (xmin2 < xmin1 < xmax2)):
        return 1

    # above
    if (ymax1 < ymin2):
        return 2

    # below
    if (ymin1 > ymax2):
        return 3

    return None


def check_prior(prior_path):
    '''
    check prior integrity
    - check that the matrix is symmetric
    - check that class1 is above class2 means class2 is below class1
    - check the 'intersect' relationship is symmetric
    '''

    pass


def inspect_prior(prior_path):
    '''
    Showing pairs with greater that 80% positional prior
    '''

    pass


# python3 positional_prior.py create /local/DEEPLEARNING/pascalvoc/VOCdevkit/VOC2007/Annotations/frcnn_trainval_ext.csv
if __name__ == '__main__':
    action = sys.argv[1]
    filepath = sys.argv[2]

    if action == 'create':
        create_positional_prior(filepath)

    elif action == 'check':
        check_prior(filepath)

    elif action == 'inspect':
        inspect_prior(filepath)
