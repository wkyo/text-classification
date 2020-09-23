FROM python:3.7.9-buster

ADD ./ /app
WORKDIR /app
RUN pip install -i https://pypi.douban.com/simple --no-cache-dir -r /app/requirements.txt && mkdir /app/instance

EXPOSE 7001
CMD [ "scripts/run_server.sh" ]