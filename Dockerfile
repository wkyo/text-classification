FROM python:3.7.9-buster

ADD ./requirements.txt /app/
WORKDIR /app
RUN pip install -i https://pypi.douban.com/simple --no-cache-dir -r /app/requirements.txt && mkdir /app/instance

ADD ./text_classification /app/text_classification
ADD ./scripts /app/scripts
ADD ./gunicorn.conf.py /app/

EXPOSE 7001
CMD [ "/bin/bash", "/app/scripts/run_server.sh" ]