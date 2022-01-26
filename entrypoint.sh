env=$1
task=$2
algo=$3
expdir=$4
reward=$5
steps=$6

if [ $# -ne 6 ]
then
	echo "illegal number of params. help: $0 <env> <task> <algo> <expdir> <reward> <steps>"
	exit -1
fi

xvfb-run -a -s "-screen 0 1400x900x24" python run_training.py --env $env --task $task --reward $reward \
                                                              --steps $steps --expdir $expdir --algo $algo
