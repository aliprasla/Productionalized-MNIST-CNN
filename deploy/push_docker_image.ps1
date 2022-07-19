# post container to registry
aws ecr-public get-login-password --region us-east-1 | docker login --username AWS --password-stdin public.ecr.aws/m0t1x7f5
docker build -t  mnist-classifier:latest .
docker tag mnist-classifier:latest public.ecr.aws/m0t1x7f5/mnist-classifier:latest
docker push public.ecr.aws/m0t1x7f5/mnist-classifier:latest
