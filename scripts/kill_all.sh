#!/bin/bash

debug_prefix="kill_all"

for i in {1..16}
do
  c_name="exp_$i"
  {
    # first try to kill single-instance experiment
    docker kill $c_name &> /dev/null && echo "[$debug_prefix] Killed: $c_name"
  } || {
    # if failed, then try to kill instances of (sequential batch) experiment
    for j in {1..10}
    do
      c_sub_name="exp_${i}_${j}"
      docker kill $c_sub_name &> /dev/null && echo "[$debug_prefix] Killed: $c_sub_name"
    done
  } || {
      # if failed again, then assume not existing container
      echo "[$debug_prefix] Failed to kill: ${c_name}_[1-10]"
  }
done