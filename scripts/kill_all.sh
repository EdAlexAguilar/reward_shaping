#!/bin/bash

debug_prefix="kill_all"

for i in {1..30}
do
  c_name="exp_$i"
  # first try to kill single-instance experiment
  docker kill $c_name &> /dev/null && echo "[$debug_prefix] Killed: $c_name"
done
