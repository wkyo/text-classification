# coding: utf-8
import multiprocessing
import os

# Log
loglevel = 'info'
accesslog = 'logs/gunicorn-access.log'
errorlog = 'logs/gunicorn-error.log'

# Server
bind = '0.0.0.0:7001'
debug = False

# Worker
# BUG: model will be loaded by each work, so high value may cause memory exhaustion
workers = 4
worker_class = 'gevent'
