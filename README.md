Required Environment Variables:

LOGLEVEL - "info"

PORT = 80

API_VERSION = "v1"


## Building and Running Docker Image Locally


```
docker build -t mnist-classifier .

//example container run
docker run -it -p 8080:8080 -e LOGLEVEL=Info -e PORT=8080 -e API_VERSION=v1 -e HOST_IP=0.0.0.0 mnist-classifier:0.1 


```

