#!/bin/bash

debug_prefix="kill_all"

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

source $DIR/init_exp_list.sh

for i in `seq 1 ${#args[@]}`
do
  c_name="exp_$i"
  # first try to kill single-instance experiment
  docker kill $c_name &> /dev/null && echo "[$debug_prefix] Killed: $c_name"
done
