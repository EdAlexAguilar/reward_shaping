env=$1
task=$2
expdir=$3
reward=$4

if [ $# -ne 4 ]
then
	echo "illegal number of params. help: $0 <env> <task> <expdir> <reward>"
	exit -1
fi

xvfb-run -s "-screen 0 1400x900x24" python run_training.py --env $env --task $task --reward $1 --steps 2000000 --expdir $expdir
