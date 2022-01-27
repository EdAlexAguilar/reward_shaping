#!/bin/bash

name=$1
env=$2
task=$3
algo=$4
expdir=$5
reward=$6
steps=$7

image=luigi/reward_shaping:pot
gpus=all

debug_prefix="run_exp"

if [ $# -ne 7 ]
then 
	echo "illegal number of params. help: $0 <exp-name> <env> <task> <algo> <exp_dir> <reward> <steps> "
	exit -1
fi

{
  docker container rm "${name}" &> /dev/null &&
  echo "[$debug_prefix] Removed existing container ${expname}"
}

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

echo "[$debug_prefix] Running ${name}"
docker run --rm -it --name $name \
	       -u $(id -u):$(id -g) -v $(pwd):/src \
	       --gpus $gpus $image \
	       /bin/bash $DIR/../entrypoint.sh $env $task $algo $expdir $reward $steps
