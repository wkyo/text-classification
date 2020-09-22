# coding: utf-8
import os
from datetime import datetime
import pickle
import glob
import logging

from .model_loader import get_model_by_name
from .model_loader import ModelLoader
from .train import train_model
from .data import load_dataset
from .keras_extends.tokenizer import JiebaTokenizer
from .utils.json_extend import dump_json


class Classifier:

    def __init__(self,
                 data_path=None,
                 data_field_label='label',
                 data_field_text='text',
                 models_path='models',
                 model_path=None):
        self.data_path = data_path
        self.data_field_label = data_field_label
        self.data_field_text = data_field_text
        self.models_path = models_path
        self.model_path = model_path

        self.tokenizer_cls = JiebaTokenizer
        self.loaded_model = None

        self.maxlen = 256

        self.load_model(model_path)

    def _gen_timed_model_name(self):
        now = datetime.now()
        return 'model-' + now.isoformat().replace(':', '')

    def _prepare_dataset(self):
        return load_dataset(
            self.data_path,
            tokenizer=self.tokenizer_cls(),
            maxlen=self.maxlen,
            field_label=self.data_field_label,
            field_text=self.data_field_text
        )

    def _train_model(self,
                     model,
                     output_dir=None,
                     train_data=None,
                     validation_data=None,
                     test_data=None,
                     data_dict=None):
        history = train_model(
            model=model,
            train_data=train_data,
            validation_data=validation_data,
            test_data=test_data,
            output_dir=output_dir
        )
        if output_dir:
            dump_json(data_dict, os.path.join(output_dir, 'data_dict.json'))

        return history

    def train(self, name):
        """Train speicified text model

        Args:
            name: the model name
        """
        output_dir = os.path.join(
            self.models_path, self._gen_timed_model_name()
        )
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        model_builder = get_model_by_name(name)
        logging.info('model <{}> loaded successfully with builder {}'.format(
            name, model_builder.__name__))

        logging.info('load dataset from {}'.format(self.data_path))
        norm_data = self._prepare_dataset()

        logging.info('build model <{}>'.format(name))
        (train_data, validation_data, test_data), data_dict = norm_data

        vocabulary_size = len(data_dict['word_index']) + 1
        n_classes = len(data_dict['classes'])
        model = model_builder(
            vocabulary_size=vocabulary_size, n_classes=n_classes)

        logging.info('train model <{}>'.format(name))
        history = self._train_model(
            model,
            output_dir=output_dir,
            train_data=train_data,
            validation_data=validation_data,
            test_data=test_data,
            data_dict=data_dict
        )

        return history

    def load_model(self, path=None):
        if not path:
            path = self.get_lastest_model()
        if not os.path.isdir(path):
            raise ValueError('model path is not a directory <{}>'.format(path))

        self.loaded_model = ModelLoader.load(
            model_path=path,
            tokenizer=self.tokenizer_cls(),
        )

    def predict(self, texts):
        return self.loaded_model.predict(texts)

    def benchmark(self):
        # TODO: run benchmarks on all models
        pass

    def get_lastest_model(self):
        model_names = glob.glob(os.path.join(self.models_path, 'model-*'))
        model_names = sorted(model_names, reverse=True)
        if model_names:
            return model_names[0]
        return None
