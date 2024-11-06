FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

# Skip timezone configuration
ENV DEBIAN_FRONTEND=noninteractive

# solution to nvidia issue: https://github.com/open-mmlab/OpenPCDet/issues/955
RUN apt-get update && apt-get install -y gnupg
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub

# Install dependencies to run on headless server and others
RUN apt-get update && apt-get install -y xvfb python-opengl \
    git swig gcc libxml2-dev libxslt1-dev zlib1g-dev g++ libfontconfig-dev \
    wget unzip
RUN pip install --upgrade "pip<=21"

WORKDIR /src
COPY . /src

# Install requirements
WORKDIR /build
COPY requirements.txt /build
COPY reward_shaping/envs/racecar/requirements.txt /build/racecar_requirements.txt
RUN pip install -r requirements.txt
RUN pip install -r racecar_requirements.txt

# Download track maps for racecar-gym
WORKDIR /build/src/racecar-gym/models/scenes
RUN wget https://github.com/luigiberducci/racecar_gym/releases/download/tracks-v2.0.0/all.zip && unzip all.zip

WORKDIR /src

