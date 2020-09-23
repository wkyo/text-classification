# coding: utf-8
import os
import functools

from flask import abort
from flask import Flask
from flask import request
from flask import jsonify
from flask_cors import CORS
import yaml

from .classifier import Classifier


def as_json(func):
    @functools.wraps(func)
    def wraper(*args, **kwargs):
        result = func(*args, **kwargs)
        return jsonify(result)

    return wraper


def create_app(instance_path='.'):
    if instance_path:
        instance_path = os.path.abspath(instance_path)
    app = Flask(__name__, instance_path=instance_path)

    if not os.path.exists(app.instance_path):
        os.makedirs(app.instance_path)

    config_path = os.path.join(instance_path, 'config.yaml')
    try:
        with open(config_path, 'rt', encoding='utf-8') as fp:
            config_data = yaml.load(fp)
            if config_data:
                app.config.from_mapping(config_data)
    except IOError:
        pass

    CORS(app)

    
    import tensorflow as tf
    # disable gpu
    if app.config.get('DISABLE_GPU', True):
        tf.config.experimental.set_visible_devices([], device_type='GPU')
    else:
        # disable gpu memory preload
        if app.config.get('DISABLE_GPU_MEMORY_PRELOAD', True):
            for gpu_dev in tf.config.list_physical_devices('GPU'):
                tf.config.experimental.set_memory_growth(gpu_dev, True)
    
    
    # load model
    models_path = os.path.join(app.instance_path, 'models')
    classifier = Classifier(model_auto_load=True, models_path=models_path)


    @app.route('/predict', methods=['POST'])
    @as_json
    # pylint: disable=unused-variable
    def predict():
        texts = request.get_json()
        if isinstance(texts, (tuple, list)):
            return classifier.predict(texts)
        abort(400, 'Invalid json format, only array is accepted.')


    @app.route('/classes', methods=['GET'])
    @as_json
    # pylint: disable=unused-variable
    def get_classes():
        return classifier.loaded_model.classes


    @app.route('/word_index', methods=['GET'])
    @as_json
    # pylint: disable=unused-variable
    def get_word_index():
        return classifier.loaded_model.word_index

    return app


def run_app(app: Flask, host='0.0.0.0', port='7001'):
    """This method is only used on development mode"""
    app.run(host=host, port=port, debug=True)
