FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-devel

RUN apt-get update && apt-get install -y xvfb python-opengl \
    git swig gcc libxml2-dev libxslt1-dev zlib1g-dev g++ libfontconfig-dev \
    wget unzip
RUN pip install --upgrade pip

WORKDIR /src
COPY . /src

# install requirements
WORKDIR /build
COPY requirements.txt /build
RUN pip install -r requirements.txt && pip install --upgrade "antlr4-python3-runtime<4.6"

WORKDIR /src

