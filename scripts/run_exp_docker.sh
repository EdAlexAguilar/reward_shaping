#!/bin/bash

reward=$1
image=luigi/reward_shaping
gpus=all

if [ $# -ne 1 ]
then 
	echo "illegal number of params. help: $0 <reward>"
	exit -1
fi

for i in 1 2 3
do 
	docker run --name exp_${reward}_${i} --rm -it -u $(id -u):$(id -g) -v $(pwd):/src --gpus $gpus $image /bin/bash entrypoint.sh $reward
done
