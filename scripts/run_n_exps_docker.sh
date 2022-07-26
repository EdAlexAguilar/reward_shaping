#!/bin/bash

expname=$1
env=$2
task=$3
algo=$4
expdir=$5
n_seeds=$6
reward=$7
steps=$8
novideo=$9

image=luigiberducci/reward_shaping:racecar
gpus="" #"--gpus all"

debug_prefix="run_n_exps"

if [ $# -ne 8 ] && [ $# -ne 9 ]
then
	echo "illegal number of params. help: $0 <exp-name> <env> <task> <algo> <exp_dir> <N-seeds> <reward> <steps> [-novideo]"
	exit -1
fi

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $DIR/..    # to mount the working dir


echo "[$debug_prefix] Running ${expname} (${n_seeds} seeds): ${env} ${task} ${algo} ${expdir} ${reward} ${steps}"

docker run --rm -it --name $expname -d \
               -u $(id -u):$(id -g) -v $(pwd):/src \
               $gpus $image \
               /bin/bash entrypoint.sh $expdir $env $task $algo $reward $steps $n_seeds $novideo
