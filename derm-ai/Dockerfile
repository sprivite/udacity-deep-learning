FROM nvidia/cuda:10.1-cudnn7-runtime as build-stage

WORKDIR /

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev gcc
    
RUN pip3 install setuptools wheel
RUN pip3 install torch jupyter fastai
RUN pip3 install sklearn

EXPOSE 5500

ENTRYPOINT jupyter-notebook --port 5500 --no-browser --ip 0.0.0.0  --allow-root
