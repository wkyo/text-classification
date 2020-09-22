# coding: utf-8
import json
from datetime import datetime

import numpy as np


class ExJsonEncoder(json.JSONEncoder):

    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, datetime):
            return o.isoformat()
        return json.JSONEncoder.default(self, o)


def load_json(path, encoding='utf-8'):
    with open(path, 'rt', encoding=encoding) as fp:
        return json.load(fp)


def dump_json(obj, path, encoding='utf-8'):
    with open(path, 'wt', encoding=encoding) as fp:
        return json.dump(obj, fp, cls=ExJsonEncoder)
