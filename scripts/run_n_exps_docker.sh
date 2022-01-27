#!/bin/bash

name=$1
env=$2
task=$3
algo=$4
expdir=$5
n_exps=$6
reward=$7
steps=$8

image=luigi/reward_shaping:pot
gpus=all

debug_prefix="run_n_exps"

if [ $# -ne 8 ]
then
	echo "illegal number of params. help: $0 <exp-name> <env> <task> <algo> <exp_dir> <N-exps> <reward> <steps>"
	exit -1
fi

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $DIR/..    # to mount the working dir

for i in `seq 1 $n_exps`
do
	expname="${name}_${i}"
	docker run --rm -it --name $expname \
               -u $(id -u):$(id -g) -v $(pwd):/src \
               --gpus $gpus $image \
               /bin/bash entrypoint.sh $env $task $algo $expdir $reward $steps

done
