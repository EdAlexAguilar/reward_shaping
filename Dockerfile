FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime

# xvbf to create a fake display and run on headless server
RUN apt-get update && apt-get install -y xvfb python-opengl \
    git swig gcc libxml2-dev libxslt1-dev zlib1g-dev g++ libfontconfig-dev
RUN pip install --upgrade pip


WORKDIR /src
COPY . /src

# install f1tenth gym
WORKDIR /src/reward_shaping/envs/f1tenth/core/
RUN git submodule init && git submodule update
WORKDIR /src/reward_shaping/envs/f1tenth/core/f1tenth_gym/gym
RUN pip install -e .

# install requirements
WORKDIR /src
RUN pip install -r requirements.txt
