# coding: utf-8
import os
import re
import glob
from datetime import datetime
import gzip
import pickle
from collections import OrderedDict
import json
import shutil

import keras
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
import numpy as np
import yaml

from .models import get_models as get_predefined_models
from .keras_extends.trainer import Trainer
from .keras_extends.tokenizer import JiebaTokenizer
from .utils.json_encoder import ExJsonEncoder

DEFAULT_ENCODING = 'utf-8'
MODEL_PREFIX = 'model-'


def is_explict_path(path):
    return re.match(r'^(\w:|\.{0,2}/|\.{0,2}\\)', path) is not None


def _smart_path(parent, subpath):
    if not parent or is_explict_path(subpath):
        return subpath
    return os.path.join(parent, subpath)


def get_lastest_model(model_dir):
    entries = [
        name
        for name in os.listdir(model_dir)
        if name.startswith(MODEL_PREFIX)
    ]
    if entries:
        entry = sorted(entries, reverse=True)[0]
        return os.path.join(model_dir, entry)
    return None


class DataLoader:
    def __init__(self, train_data=None, validation_data=None, test_data=None, data_dict=None):
        self.train_data = train_data
        self.validation_data = validation_data
        self.test_data = test_data
        self.data_dict = data_dict

    def __call__(self):
        return self.train_data, self.validation_data, self.test_data


class GzPickleLoader(DataLoader):

    @classmethod
    def load(cls, path, fill_validation=True):
        with gzip.open(path, 'rb') as fp:
            data = pickle.load(fp)

        data_dict = data[-1]
        if len(data) == 3:
            train_data, validation_data, test_data = data[0], None, data[1]
        elif len(data) == 4:
            train_data, validation_data, test_data = data[: 3]
        else:
            raise ValueError('Unsupported pickle dataset')

        if validation_data is None:
            test_size = len(test_data[0])
            permutes = np.random.permutation(test_size)[: int(test_size * 0.5)]
            validation_data = test_data[0][permutes], test_data[1][permutes]
        return cls(train_data, validation_data, test_data, data_dict)


class TextCategoricalModel:

    def __init__(self, model_path=None):
        if model_path:
            self.init_model(model_path)

    def _load_tokenizer(self):
        if self.lang.lower() in ('zh', 'cn', 'zh_cn'):
            tokenizer = JiebaTokenizer()
        else:
            tokenizer = Tokenizer()

        tokenizer.word_index = self.word_index
        return tokenizer

    def init_model(self, model_path):
        self.model_path = model_path
        self.best_model_path = os.path.join(model_path, 'model.h5')
        self.meta_path = os.path.join(model_path, 'meta.json')

        self.model = keras.models.load_model(self.best_model_path)
        with open(os.path.join(model_path, 'meta.json'), 'rt') as fp:
            self.meta_info = json.load(fp)

        self.tokenizer = self._load_tokenizer()

    def predict(self, texts):
        sequences = self.tokenizer.texts_to_sequences(texts)
        sequences = pad_sequences(sequences)
        y_true = self.model.predict(sequences)
        indices = np.argmax(y_true, axis=-1)
        labels = [self.classes[idx] for idx in indices]
        return indices, labels

    def __call__(self, texts):
        return self.predict(texts)[-1]

    def __bool__(self):
        return True

    @property
    def lang(self):
        return self.meta_info['lang']

    @property
    def word_index(self):
        return self.meta_info['word_index']

    @property
    def vocabulary_size(self):
        return len(self.meta_info['word_index']) + 1

    @property
    def classes(self):
        return self.meta_info['classes']


class TextClassifier:

    def __init__(self,
                 proj_dir='.',
                 model_dir='models',
                 data_dir='data',
                 history_size=5,
                 config_path='config.yml',
                 lang=None,
                 development=True):

        self.proj_dir = proj_dir
        self.model_dir = _smart_path(proj_dir, model_dir)
        self.data_dir = _smart_path(proj_dir, data_dir)
        self.config_path = _smart_path(proj_dir, config_path)
        self.lang = lang
        self.history_size = history_size

        with open(self.config_path, 'rt', encoding=DEFAULT_ENCODING) as fp:
            self.settings = yaml.load(fp)

        if not self.lang:
            self.lang = self.settings.get('lang', 'cn')

        self.model = None
        self.trainer = None

    def load_config(self):
        with open(self.config_path, 'rt', encoding=DEFAULT_ENCODING) as fp:
            settings = yaml.load(fp)
        return settings

    @property
    def data_loader(self):
        attr_name = '__lazy_data_loader'
        if not hasattr(self, attr_name):
            dataset_path = _smart_path(
                self.data_dir, self.settings['data']['dataset'])
            if dataset_path.endswith('.pickle.gz'):
                loader = GzPickleLoader.load(dataset_path)
                setattr(self, attr_name, loader)
            else:
                raise ValueError('Unsupported dataset type')
        return getattr(self, attr_name)

    def train(self, name='TextCNN'):
        model_info = None
        for info in get_predefined_models():
            if name == info['name']:
                model_info = info
                break
        else:
            raise ValueError('Mode {} not found'.format(name))
        model_builder = model_info['build']

        now = datetime.now()

        model_name = MODEL_PREFIX + now.isoformat().replace(':', '')
        model_path = os.path.join(self.model_dir, model_name)
        os.mkdir(model_path)

        vocabulary_size = len(self.data_loader.data_dict['word_index']) + 1
        n_classes = len(self.data_loader.data_dict['classes'])
        model = model_builder(
            vocabulary_size=vocabulary_size, n_classes=n_classes)

        trainer = Trainer(model_dir=model_path,
                          data_loader=self.data_loader)
        history = trainer(model)

        meta_info = OrderedDict([
            ('name', name),
            ('date', now.isoformat()),
            ('lang', self.lang),
            ('classes', self.data_loader.data_dict['classes']),
            ('word_index', self.data_loader.data_dict['word_index']),
            ('history', history),
        ])

        # write meta info
        with open(os.path.join(model_path, 'meta.json'), 'wt') as fp:
            json.dump(meta_info, fp, cls=ExJsonEncoder)

    def load_model(self, path=None):
        if path is None:
            path = get_lastest_model(self.model_dir)
        else:
            path = _smart_path(self.model_dir, path)
        self.model = TextCategoricalModel(path)

    def predict(self, texts):
        if not self.model:
            raise RuntimeError('Model has not been loaded')
        return self.model(texts)

    def get_models(self):
        entries = [
            name
            for name in os.listdir(self.model_dir)
            if name.startswith(MODEL_PREFIX)
        ]
        entries = sorted(entries, reverse=True)
        return entries

    def clear_old_models(self):
        if self.history_size > 0:
            models = self.get_models()[self.history_size:]
            for item in models:
                os.remove(os.path.join(self.model_dir, item))
