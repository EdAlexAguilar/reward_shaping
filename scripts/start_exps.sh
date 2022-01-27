#!/bin/bash

logdir=$(date '+%d%m%Y_%H%M%S')

debug_prefix="start_exp"

# sequential-run arguments: <env> <task> <algo> <exp_dir> <N-exps> <reward> <steps>
# individual-run arguments: <env> <task> <algo> <exp_dir> <reward> <steps>
args=(
  #
  # Sequential run of 3 seeds, 1M steps for easy envs: cpole and bw
  #
  "cart_pole_obst fixed_height sac ${logdir} 3 default 1000000"
  "cart_pole_obst fixed_height sac ${logdir} 3 hrs_pot 1000000"
  "bipedal_walker forward sac ${logdir} 3 default 1000000"
  "bipedal_walker forward sac ${logdir} 3 hrs_pot 1000000"
  #
  # Parallel run of 3 seeds, 2M steps for bw hardcore
  #
  "bipedal_walker forward_hardcore sac ${logdir} default 2000000"
  "bipedal_walker forward_hardcore sac ${logdir} default 2000000"
  "bipedal_walker forward_hardcore sac ${logdir} default 2000000"
  "bipedal_walker forward_hardcore sac ${logdir} hrs_pot 2000000"
  "bipedal_walker forward_hardcore sac ${logdir} hrs_pot 2000000"
  "bipedal_walker forward_hardcore sac ${logdir} hrs_pot 2000000"
  #
  # Parallel run of 3 seeds, 2M steps for lunar lander
  #
  "lunar_lander land  sac ${logdir} default 2000000"
  "lunar_lander land  sac ${logdir} default 2000000"
  "lunar_lander land  sac ${logdir} default 2000000"
  "lunar_lander land sac ${logdir} hrs_pot 2000000"
  "lunar_lander land sac ${logdir} hrs_pot 2000000"
  "lunar_lander land sac ${logdir} hrs_pot 2000000"
)

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

for exp in "$@"
do
  index=$(($exp-1))
  n_args=$(echo ${args[$index]} | wc -w)
  #
  if [ $n_args == 7 ]; then
      $DIR/run_n_exps_docker.sh exp_$exp $(echo ${args[$index]})
  elif [ $n_args == 6 ]; then
      $DIR/run_exp_docker.sh exp_$exp $(echo ${args[$index]})
  else
      echo "[$debug_prefix] Invalid exp: exp_$exp"
  fi
done
