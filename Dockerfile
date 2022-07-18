FROM python:3.10-slim

WORKDIR /usr/local/src

COPY requirements.txt requirements.txt

RUN apt-get update \
    && pip install --no-cache-dir -r requirements.txt 

EXPOSE $PORT

COPY . .

RUN mkdir data/models/

RUN python3 src/run_model_training.py

ENTRYPOINT ["python3","src/run_api.py"]


