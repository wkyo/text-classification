# coding: utf-8
import os
import importlib
import logging

import keras
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
import numpy as np

from .utils.json_extend import load_json

WORK_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(WORK_DIR, 'models')

MODEL_CACHE = {}


def scan_models():
    models = []
    for name in os.listdir(MODELS_DIR):
        if name.startswith('_') or name.startswith('.') or not name.endswith('.py'):
            continue
        models.append(name[: -3])
    return models


def get_model_by_name(name):
    if name not in MODEL_CACHE:
        module_path = os.path.join(MODELS_DIR, name + '.py')
        spec = importlib.util.spec_from_file_location(
            '_models_' + name, module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        MODEL_CACHE[name] = module.EXPORTS['build']
    return MODEL_CACHE[name]


class ModelLoader:

    def __init__(self,
                 model_path=None,
                 tokenizer=None,
                 tokenizer_update=True):
        if model_path:
            self.init_model(
                model_path,
                tokenizer=tokenizer,
                tokenizer_update=tokenizer_update
            )

    def init_model(self, model_path, tokenizer=None, tokenizer_update=True):
        self.model_path = model_path

        best_model_path = os.path.join(model_path, 'model.h5')
        self.model = keras.models.load_model(best_model_path)

        data_dict_path = os.path.join(model_path, 'data_dict.json')
        self.data_dict = load_json(data_dict_path)

        if not tokenizer:
            tokenizer = Tokenizer()
        if tokenizer_update:
            data_dict_path = os.path.join(model_path, 'data_dict.json')
            tokenizer.word_index = self.data_dict['word_index']
        self.tokenizer = tokenizer

    def predict(self, texts, maxlen=256):
        # BUG: for model based on convolution, the maxlen must larger than kernel size
        sequences = self.tokenizer.texts_to_sequences(texts)
        sequences = pad_sequences(sequences, maxlen=maxlen)
        y_pred = self.model.predict(sequences)
        y_indices = np.argmax(y_pred, axis=-1)
        labels = [self.classes[idx] for idx in y_indices]
        return labels

    @property
    def classes(self):
        return self.data_dict['classes']

    @classmethod
    def load(cls,
             model_path=None,
             tokenizer=None,
             tokenizer_update=True):
        return cls(
            model_path,
            tokenizer=tokenizer,
            tokenizer_update=tokenizer_update
        )
