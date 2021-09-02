env=$1
task=$2
algo=$3
expdir=$4
reward=$5

if [ $# -ne 5 ]
then
	echo "illegal number of params. help: $0 <env> <task> <algo> <expdir> <reward>"
	exit -1
fi

xvfb-run -a -s "-screen 0 1400x900x24" python run_training.py --env $env --task $task --reward $reward \
                                                              --steps 2000000 --expdir $expdir --algo $algo
