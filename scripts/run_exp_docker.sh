#!/bin/bash

env=$1
task=$2
algo=$3
expdir=$4
reward=$5
steps=$6

image=luigi/reward_shaping:pot
gpus=all

if [ $# -ne 6 ]
then 
	echo "illegal number of params. help: $0 <env> <task> <algo> <exp_dir> <reward> <steps> "
	exit -1
fi

docker run --rm -it \
	       -u $(id -u):$(id -g) -v $(pwd):/src \
	       --gpus $gpus $image \
	       /bin/bash entrypoint.sh $env $task $algo $expdir $reward $steps
