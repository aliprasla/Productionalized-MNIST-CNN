FROM python:3.11-rc-slim

WORKDIR /usr/local/src

COPY requirements.txt requirements.txt

RUN apt-get update \
    && pip install --no-cache-dir -r requirements.txt 

EXPOSE $PORT

COPY . .

ENTRYPOINT ["python3","src/run_api.py"]
