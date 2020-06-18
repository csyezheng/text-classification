"""
Read the required fields (texts and labels).
Do any pre-processing if required. For example, make sure all label values are in range [0, num_classes-1].
Split the data into training, validation sets and testing sets.
Shuffle the training data.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random

import numpy as np
import jieba


def read_category():
    categories = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']
    cat_to_id = dict(zip(categories, range(len(categories))))
    return categories, cat_to_id

def stop_words(data_path):
    filename = os.path.join(data_path, 'stopwords.txt')
    return [line.strip() for line in open(filename).readlines()]

def text_segmentation(text):
    return " ".join(jieba.cut(text, cut_all=False))

def shuffle_list(*ls):
    l = list(zip(*ls))
    random.shuffle(l)
    return zip(*l)

def load_cnews_dataset(data_path, seed=123):
    """Loads the cnews dataset.

    # Arguments
        data_path: string, path to the data directory.
        seed: int, seed for randomizer.

    # Returns
        A tuple of training and validation data.

        Download and uncompress archive from:
        http://thuctc.thunlp.org/

        Alternative subset can be download from: https://pan.baidu.com/s/1hugrfRu 
        password: qfud
    """

    categories, cat_to_id = read_category()

    train_data_path = os.path.join(data_path, 'cnews.train.txt')
    val_data_path = os.path.join(data_path, 'cnews.val.txt')
    test_data_path = os.path.join(data_path, 'cnews.test.txt')

    # Load the training data
    train_texts = []
    train_labels = []
    with open(train_data_path) as f:
        for line in f.readlines():
            label, raw_text = line.split('\t')
            train_texts.append(text_segmentation(raw_text))
            train_labels.append(cat_to_id[label])


    # Load the validation data.
    val_texts = []
    val_labels = []
    with open(val_data_path) as f:
        for line in f.readlines():
            label, raw_text = line.split('\t')
            val_texts.append(text_segmentation(raw_text))
            val_labels.append(cat_to_id[label])

    # Load the test data.
    test_texts = []
    test_labels = []
    with open(test_data_path) as f:
        for line in f.readlines():
            label, raw_text = line.split('\t')
            test_texts.append(text_segmentation(raw_text))
            test_labels.append(cat_to_id[label])

    # Verify that validation labels are in the same range as training labels.
    unexpected_labels_id = [v for v in val_labels if v not in train_labels]
    if len(unexpected_labels_id):
        raise ValueError('Unexpected label values found in the validation set:'
                         ' {unexpected_labels}. Please make sure that the '
                         'labels in the validation set are in the same range '
                         'as training labels.'.format(
                             unexpected_labels=", ".join([
                                 categories[id] for id in unexpected_labels_id])))

    train_texts, train_labels = shuffle_list(train_texts, train_labels)
    val_texts, val_labels = shuffle_list(val_texts, val_labels)
    test_texts, test_labels = shuffle_list(test_texts, test_labels)
    return ((train_texts, np.array(train_labels)),
            (val_texts, np.array(val_labels)),
            (test_texts, np.array(test_labels)))