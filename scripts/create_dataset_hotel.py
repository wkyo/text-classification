#!/usr/bin/env python3
# coding: utf-8
import os
import pickle
import gzip
import csv

import numpy as np
from sklearn.model_selection import train_test_split
from keras_preprocessing.sequence import pad_sequences


def load_hotel(path):
    from text_classification.keras_extends.tokenizer import JiebaTokenizer
    
    pos_data, neg_data = [], []

    tokenizer = JiebaTokenizer()

    with open(path, 'rt', encoding='utf-8') as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            label = int(row['label'])
            if label == 1:
                pos_data.append(row['review'])
            else:
                neg_data.append(row['review'])
    
            tokenizer.fit_on_texts([row['review']])

    pos_data = tokenizer.texts_to_sequences(pos_data)
    neg_data = tokenizer.texts_to_sequences(neg_data)

    pos_data = pad_sequences(pos_data, maxlen=256)
    neg_data = pad_sequences(neg_data, maxlen=256)

    pos_train, pos_test = train_test_split(pos_data, test_size=0.2)
    neg_train, neg_test = train_test_split(neg_data, test_size=0.2)

    x_train = np.concatenate((pos_train, neg_train), axis=0)
    y_train = np.concatenate((np.ones(len(pos_train)), np.zeros(len(neg_train))))

    x_test = np.concatenate((pos_test, neg_test), axis=0)
    y_test = np.concatenate((np.ones(len(pos_test)), np.zeros(len(neg_test))))

    data_dict = {
        'word_counts': tokenizer.word_counts,
        'word_index': tokenizer.word_index,
        'classes': ['neg', 'pos']
    }

    return (x_train, y_train), (x_test, y_test), data_dict


if __name__ == "__main__":
    root_dir = os.path.join(os.path.dirname(__file__), '..')

    import sys

    sys.path.append(root_dir)

    dataset_dir = os.path.join(root_dir, 'dataset')
    dataset = load_hotel(os.path.join(dataset_dir, 'zh_hotel_review.csv'))
    with gzip.open('zh_hotel_review.pickle.gz', 'wb') as fp:
        pickle.dump(dataset, fp)
        