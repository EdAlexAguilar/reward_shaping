expdir=$1
env=$2
task=$3
algo=$4
reward=$5
steps=$6
n_seeds=$7
novideo=$8

if [ $# -ne 7 ] && [ $# -ne 8 ]
then
	echo "illegal number of params. help: $0 <expdir> <env> <task> <algo> <reward> <steps> <n_seeds> [-no_video]"
	exit -1
fi

xvfb-run -a -s "-screen 0 1400x900x24" python run_training.py --env $env --task $task --reward $reward \
                                                              --steps $steps --expdir $expdir --algo $algo \
                                                              --n_seeds $n_seeds $novideo
