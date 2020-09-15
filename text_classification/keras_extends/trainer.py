# coding: utf-8
import os
from datetime import datetime
import json

from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from keras import Model

from .callbacks import F1Score, HistoryLogger


class Trainer:

    def __init__(self,
                 model_dir=None,
                 early_stop=True,
                 batch_size=128,
                 epochs=20,
                 save_best_only=True,
                 monitor='val_accuracy',
                 with_tensorboard=True,
                 data_loader=None):
        self.model_dir = model_dir

        if isinstance(early_stop, bool):
            if early_stop:
                early_stop = 5
            else:
                early_stop = -1
        self.early_stop = early_stop

        self.batch_size = batch_size
        self.epochs = epochs
        self.save_best_only = save_best_only
        self.monitor = monitor
        self.with_tensorboard = with_tensorboard
        self.train_data, self.validation_data, self.test_data = data_loader()

    def prepare_train_envs(self):
        callbacks = []

        if self.early_stop >= 0:
            callbacks.append(EarlyStopping(
                monitor=self.monitor, patience=self.early_stop))

        if self.validation_data:
            callbacks.append(F1Score(self.validation_data))

        model_dir = self.model_dir

        if model_dir:
            history_path = os.path.join(model_dir, 'history.json')
            callbacks.append(HistoryLogger(history_path))

            model_path = os.path.join(model_dir, 'model.h5')
            callbacks.append(ModelCheckpoint(
                model_path, monitor=self.monitor, save_best_only=self.save_best_only))

            if self.with_tensorboard:
                tensorboard_path = os.path.join(model_dir, 'logs')
                if not os.path.exists(tensorboard_path):
                    os.mkdir(tensorboard_path)
                callbacks.append(TensorBoard(log_dir=tensorboard_path))

        self.callbacks = callbacks

    def __call__(self, model):
        self.prepare_train_envs()

        if model.compiled_loss is None:
            model.compile(optimizer='adam', metrics=[
                          'accuracy'], loss='sparse_categorical_crossentropy')

        model.summary()

        x, y = self.train_data
        history = model.fit(
            x, y, batch_size=self.batch_size, epochs=self.epochs,
            validation_data=self.validation_data, callbacks=self.callbacks)

        metrics = model.evaluate(*self.test_data)

        stat = {
            'train': history.history,
            'test': {k: v for k, v in zip(model.metrics_names, metrics)}
        }

        return stat
