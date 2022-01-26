#!/bin/bash

env=$1
task=$2
algo=$3
expdir=$4
n_exps=$5
reward=$6
steps=$7

image=luigi/reward_shaping:pot
gpus=all

if [ $# -ne 7 ]
then
	echo "illegal number of params. help: $0 <env> <task> <algo> <exp_dir> <N-exps> <reward> <steps>"
	exit -1
fi

for i in `seq 1 $n_exps`
do
	./scripts/run_exp_docker.sh $env $task $algo $expdir $reward $steps
done
