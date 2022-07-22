#!/bin/bash

logdir=$(date '+%d%m%Y_%H%M%S')

debug_prefix="start_exp"

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

source $DIR/init_exp_list.sh

for exp in "$@"
do
  index=$(($exp-1))
  n_args=$(echo ${args[$index]} | wc -w)
  #
  if [ $n_args == 7 ] || [ $n_args == 8 ]; then
	  $DIR/run_n_exps_docker.sh exp_$exp $(echo ${args[$index]})
  else
	  echo "[$debug_prefix] Invalid exp: exp_$exp"
  fi
done
