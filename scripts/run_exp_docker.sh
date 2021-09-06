#!/bin/bash

env=$1
task=$2
algo=$3
expdir=$4
reward=$5

image=luigi/reward_shaping
gpus=all

if [ $# -ne 5 ]
then 
	echo "illegal number of params. help: $0 <env> <task> <algo> <exp_dir> <reward> "
	exit -1
fi

docker run --rm -it \
	       -u $(id -u):$(id -g) -v $(pwd):/src \
	       $image \
	       /bin/bash entrypoint.sh $env $task $algo $expdir $reward
