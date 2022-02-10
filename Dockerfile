FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime

# xvbf to create a fake display and run on headless server
RUN apt-get update && apt-get install -y xvfb python-opengl \
    git swig gcc libxml2-dev libxslt1-dev zlib1g-dev g++ libfontconfig-dev \
    wget unzip
RUN pip install --upgrade pip

WORKDIR /src
COPY . /src

# install requirements
WORKDIR /build
COPY requirements.txt /build
RUN pip install -r requirements.txt

# download track maps
WORKDIR /build/src/racecar-gym/models/scenes
RUN wget https://github.com/axelbr/racecar_gym/releases/download/tracks-v1.0.0/all.zip && unzip all.zip

# install f1tenth gym
WORKDIR /src/reward_shaping/envs/f1tenth/core/f1tenth_gym/gym
RUN pip install -e .

WORKDIR /src

