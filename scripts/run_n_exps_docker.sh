#!/bin/bash

env=$1
task=$2
algo=$3
expdir=$4
n_exps=$5
reward=$6

image=luigi/reward_shaping
gpus=all

if [ $# -ne 6 ]
then
	echo "illegal number of params. help: $0 <env> <task> <algo> <exp_dir> <N-exps> <reward> "
	exit -1
fi

for i in `seq 1 $n_exps`
do
	./scripts/run_exp_docker.sh $env $task $expdir $reward
done
