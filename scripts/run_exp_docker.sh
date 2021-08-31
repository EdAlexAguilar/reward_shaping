#!/bin/bash

env=$1
task=$2
expdir=$3
reward=$4
image=luigi/reward_shaping
gpus=all

if [ $# -ne 4 ]
then 
	echo "illegal number of params. help: $0 <env> <task> <exp_dir> <reward> "
	exit -1
fi

docker run --name exp_${env}_${reward}_${i} --rm -it \
	       -u $(id -u):$(id -g) -v $(pwd):/src \
	       --gpus $gpus $image \
	       /bin/bash entrypoint.sh $env $task $expdir $reward
