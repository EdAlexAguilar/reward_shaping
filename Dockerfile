FROM pytorch/pytorch:latest

# xvbf to create a fake display and run on headless server
RUN apt-get update && apt-get install -y xvfb python-opengl \
    git swig gcc libxml2-dev libxslt1-dev zlib1g-dev g++ libfontconfig-dev

# install requirements
WORKDIR /src
COPY . /src
RUN pip install -r requirements.txt