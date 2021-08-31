#!/bin/bash

env=$1
task=$2
expdir=$3
n_exps=$4
reward=$5
image=luigi/reward_shaping
gpus=all

if [ $# -ne 5 ]
then
	echo "illegal number of params. help: $0 <env> <task> <exp_dir> <N-exps> <reward> "
	exit -1
fi

for i in `seq 1 $n_exps`
do
	./scripts/run_exp_docker.sh $env $task $expdir $reward
done
