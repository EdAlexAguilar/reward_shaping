FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel

# solution to nvidia issue: https://github.com/open-mmlab/OpenPCDet/issues/955
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub

RUN apt-get update && apt-get install -y xvfb python-opengl \
    git swig gcc libxml2-dev libxslt1-dev zlib1g-dev g++ libfontconfig-dev \
    unzip wget
RUN pip install --upgrade pip

WORKDIR /src
COPY . /src

# install requirements
WORKDIR /build
COPY requirements.txt /build
COPY reward_shaping/envs/racecar/requirements.txt /build/racecar_requirements.txt
RUN pip install -r requirements.txt && pip install --upgrade "antlr4-python3-runtime<4.6"
RUN pip install -r racecar_requirements.txt

WORKDIR /src

