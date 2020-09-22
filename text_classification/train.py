# coding: utf-8
import os
import json

import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.callbacks import EarlyStopping

from .keras_extends.callbacks import F1Score
from .keras_extends.callbacks import HistoryLogger
from .utils.json_extend import dump_json as _dump_json


def train_model(model,
                train_data=None,
                validation_data=None,
                test_data=None,
                output_dir=None,
                epochs=25,
                batch_size=128,
                monitor='val_accuracy'):
    # check keras model is compiled
    if not getattr(model, 'compiled_loss'):
        model.compile(
            optimizer='adam',
            metrics=['accuracy'],
            loss='sparse_categorical_crossentropy'
        )

    if validation_data is None:
        validation_data = test_data
    x, y = train_data

    # make callbacks
    callbacks = []
    if output_dir:
        callbacks.append(ModelCheckpoint(
            os.path.join(output_dir, 'model.h5'),
            monitor=monitor,
            save_best_only=True,
        ))
        callbacks.append(TensorBoard(
            log_dir=os.path.join(output_dir, 'logs')
        ))

        if validation_data:
            callbacks.append(HistoryLogger(
                os.path.join(output_dir, 'train.json')
            ))

    if validation_data:
        callbacks.append(F1Score(
            validation_data
        ))

    callbacks.append(EarlyStopping(
        monitor=monitor,
        patience=5
    ))

    history = {}

    history['train'] = model.fit(
        x, y, batch_size=batch_size, epochs=epochs,
        validation_data=validation_data, callbacks=callbacks
    ).history

    # evaluate model
    if test_data:
        metrics = model.evaluate(*test_data)
        history['evaluate'] = {k: v for k,
                               v in zip(model.metrics_names, metrics)}

    # write history as json
    if output_dir:
        _dump_json(history, os.path.join(output_dir, 'history.json'))
    
    return history
