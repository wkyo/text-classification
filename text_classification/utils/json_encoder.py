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