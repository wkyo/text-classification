import fnmatch
import os
import re
import glob
import importlib


MODELS = []


def __scan_models(path):
    model_info = []
    for entry in glob.glob(os.path.join(path, '*.py')):
        # ignore hidden python modules
        basename = os.path.basename(entry)
        if basename.startswith('_'):
            continue
        
        spec = importlib.util.spec_from_file_location('_inter_models_' + basename[:-3], entry)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        model_info.append(module.EXPORTS)
    return model_info


def get_models():
    if not MODELS:
        path = os.path.dirname(__file__)
        model_info = __scan_models(path)
        MODELS.extend(model_info)

    return MODELS
