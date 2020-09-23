#!/bin/bash
set -e

INSTANCE_PATH=/app/instance

export PYTHONPATH=/app:$PYTHONPATH

cd $INSTANCE_PATH && gunicorn -c /app/gunicorn.conf.py 'text_classification:create_app()'