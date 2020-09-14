# coding: utf-8
import importlib
import os
import gzip
import pickle
import json

import numpy as np
from sklearn.model_selection import train_test_split

from .models import get_models
from .utils.camelize import uncamelize
from .utils.json_encoder import ExJsonEncoder


__version__ = '0.1'
SUPPORTED_MODEL_IMPLS = [
    'keras',
    'tensorflow',
    'pytorch'
]
PKG_DIR = os.path.dirname(__file__)


def cmd_show_models(backend=None):
    print('{:10s} {:18s}'.format('backend', 'name'))
    print('{:10s} {:18s}'.format('---', '---'))

    for model in get_models():
        print('{:10s} {:18s}'.format(model['backend'], model['name']))


def cmd_train_model(model_name, dataset='imdb_mini', export_dir='build/models', lang='en'):
    from .keras_extends.trainer import ModelTrainer

    if not os.path.exists(export_dir):
        os.makedirs(export_dir)

    model_info = None
    for info in get_models():
        if model_name == info['name']:
            model_info = info
            break
    else:
        raise ValueError('Mode {} not found'.format(model_name))

    model_builder = model_info['build']

    # prepare dataset: train, validation, test
    dataset_path = os.path.join(
        os.path.dirname(__file__), 'data', dataset + '.pickle.gz')
    with gzip.open(dataset_path, 'rb') as fp:
        train_data, test_data, data_dict = pickle.load(fp)
    print(train_data[0].shape)
    print(train_data[1].shape)

    x_train, x_val, y_train, y_val = train_test_split(
        train_data[0], train_data[1], train_size=0.8)
    # x_train, x_val, y_train, y_val = result
    train_data, validation_data = (x_train, y_train), (x_val, y_val)

    vocabulary_size = len(data_dict['word_index']) + 1
    n_classes = len(data_dict['classes'])

    model = model_builder(
        vocabulary_size=vocabulary_size,
        n_classes=n_classes,
        name=model_info['name']
    )

    def data_loader(): return (train_data, validation_data, test_data)

    trainer = ModelTrainer(export_dir=export_dir, data_loader=data_loader)
    history = trainer(model)
    history['name'] = model_info['name']
    history['dataset'] = dataset
    history['lang'] = lang

    with open(os.path.join(trainer.model_dir, 'meta.json'), 'wt', encoding='utf-8') as fp:
        json.dump(history, fp, cls=ExJsonEncoder)

    with open(os.path.join(trainer.model_dir, 'data_dict.json'), 'wt', encoding='utf-8') as fp:
        json.dump(data_dict, fp, cls=ExJsonEncoder)


def cmd_predict(model_dir, texts):
    if not os.path.exists(model_dir):
        raise IOError('model not exists')

    with open(os.path.join(model_dir, 'data_dict.json'), encoding='utf-8') as fp:
        data_dict = json.load(fp)
    
    with open(os.path.join(model_dir, 'meta.json'), encoding='utf-8') as fp:
        meta_info = json.load(fp)

    if meta_info['lang'] == 'zh':
        from .keras_extends.tokenizer import JiebaTokenizer as Tokenizer
    else:
        from keras_preprocessing.text import Tokenizer
    from keras_preprocessing.sequence import pad_sequences

    tokenizer = Tokenizer()
    tokenizer.word_counts = data_dict['word_counts']
    tokenizer.word_index = data_dict['word_index']
    sequences = tokenizer.texts_to_sequences(texts)

    sequences = pad_sequences(sequences, maxlen=256)

    import keras
    model = keras.models.load_model(os.path.join(model_dir, 'model.h5'))
    y_pred = model.predict(sequences)
    y_pred = np.argmax(y_pred, axis=-1)
    classes = data_dict['classes']
    y_pred = [classes[cls_idx] for cls_idx in y_pred]

    for idx, class_ in enumerate(y_pred):
        print('{:>5d} {}'.format(idx, class_))


def main():
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parsers = parser.add_subparsers(dest='cmd', help='subcommand')

    subparser_ls = parsers.add_parser('ls')
    subparser_ls.add_argument('--backend', default='keras')

    subparser_train = parsers.add_parser('train')
    subparser_train.add_argument(
        '--early-stop', default=False, action='store_true')
    subparser_train.add_argument('--dataset', default=None)
    subparser_train.add_argument('--epochs', default=15, type=int)
    subparser_train.add_argument('--batch-size', default=128, type=int)
    subparser_train.add_argument('--export-dir', default='build/models')
    subparser_train.add_argument('--lang', default='en', choices=['zh', 'en'])
    subparser_train.add_argument('model_name')

    subparser_predict = parsers.add_parser('predict')
    subparser_predict.add_argument('--model', default=None)
    subparser_predict.add_argument('texts', nargs='+')

    args = parser.parse_args()

    if args.cmd == 'ls':
        cmd_show_models()
    elif args.cmd == 'train':
        if args.lang == 'en':
            if args.dataset is None:
                args.dataset = 'imdb_mini'
        elif args.lang == 'zh':
            if args.dataset is None:
                args.dataset = 'zh_hotel_review'
        else:
            raise NotImplementedError
        cmd_train_model(
            args.model_name,
            dataset=args.dataset,
            export_dir=args.export_dir,
            lang=args.lang
        )
    elif args.cmd == 'predict':
        cmd_predict(args.model, args.texts)
