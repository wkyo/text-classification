#!/usr/bin/env python3
# coding: utf-8
import os
import pickle
import gzip

import numpy as np
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder


def read_cat_data(path, limit=None):
    items = [name for name in sorted(
        os.listdir(path)) if not name.startswith('.')]
    if limit:
        items = items[: limit]

    for name in items:
        text_path = os.path.join(path, name)
        with open(text_path, 'rt', encoding='utf-8') as fp:
            content = fp.read().strip()
            yield content


def make_imdb_dataset_mini(unpacked_imdb_dir):
    data_divides = (
        ('train', 'pos', 2000),
        ('train', 'neg', 2000),
        ('test', 'pos', 500),
        ('test', 'neg', 500),
    )

    tokenizer = Tokenizer()
    label_encoder = LabelEncoder()

    label_encoder.fit([x[1] for x in data_divides])

    # generate word index from corpus
    for type_, label, limit in data_divides:
        data_dir = os.path.join(unpacked_imdb_dir, type_, label)
        tokenizer.fit_on_texts(read_cat_data(data_dir, limit))

    x_train, y_train, x_test, y_test = [], [], [], []
    for type_, label, limit in data_divides:
        data_dir = os.path.join(unpacked_imdb_dir, type_, label)
        label_idx = label_encoder.transform([label])[0]
        sequences = tokenizer.texts_to_sequences(
            read_cat_data(data_dir, limit))
        if type_ == 'train':
            x_train.extend(sequences)
            y_train.extend([label_idx for _ in sequences])
        else:
            x_test.extend(sequences)
            y_test.extend([label_idx for _ in sequences])

    x_train = pad_sequences(x_train, maxlen=200)
    y_train = np.asarray(y_train)

    x_test = pad_sequences(x_test, maxlen=200)
    y_test = np.asarray(y_test)

    data_dict = {
        'word_counts': tokenizer.word_counts,
        'word_index': tokenizer.word_index,
        'classes': label_encoder.classes_
    }

    return (x_train, y_train), (x_test, y_test), data_dict


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="""
Please download dataset manually from `http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz`, and unpack it to somewhere
    """)
    parser.add_argument('-o', '--output', help='file name without suffix')
    parser.add_argument('imdb', help='unpacked imdb dataset')
    args = parser.parse_args()

    target = args.output + '.pickle.gz'
    data = make_imdb_dataset_mini(args.imdb)
    with gzip.open(target, 'wb') as fp:
        pickle.dump(data, fp)
