#!/bin/bash

debug_prefix="kill_all"

for i in {1..16}
do
  c_name="exp_$i"
  {
    docker kill $c_name &> /dev/null && echo "[$debug_prefix] Killed: $c_name"
  } || {
    echo "[$debug_prefix] Failed to kill: $c_name"
  }
done