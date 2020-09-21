# coding: utf-8
import os
import importlib

WORK_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(WORK_DIR, 'models')

MODEL_CACHE = {}


def scan_models():
    models = []
    for name in os.listdir(MODELS_DIR):
        if name.startswith('_') or name.startswith('.') or not name.endswith('.py'):
            continue
        models.append(name)
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
