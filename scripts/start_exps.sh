#!/bin/bash

logdir=$(date '+%d%m%Y_%H%M%S')

debug_prefix="start_exp"

# sequential-run arguments: <env> <task> <algo> <exp_dir> <N-exps> <reward> <steps>
args=(
  ##
  ## Cartpole
  ##
  "cart_pole_obst fixed_height sac ${logdir} 1 default 1000000"
  "cart_pole_obst fixed_height sac ${logdir} 1 hrs_pot 1000000"
  "cart_pole_obst fixed_height sac ${logdir} 1 tltl 1000000"
  "cart_pole_obst fixed_height sac ${logdir} 1 bhnr 1000000"
  "cart_pole_obst fixed_height sac ${logdir} 1 morl_uni 1000000"
  "cart_pole_obst fixed_height sac ${logdir} 1 morl_dec 1000000"
  ##
  ## Bipedal Walker
  ##
  "bipedal_walker forward sac ${logdir} 1 default 1000000"
  "bipedal_walker forward sac ${logdir} 1 hrs_pot 1000000"
  "bipedal_walker forward sac ${logdir} 1 tltl 1000000"
  "bipedal_walker forward sac ${logdir} 1 bhnr 1000000"
  "bipedal_walker forward sac ${logdir} 1 morl_uni 1000000"
  "bipedal_walker forward sac ${logdir} 1 morl_dec 1000000"
  ##
  ## Bipedal Walker Hardcore
  ##
  "bipedal_walker forward_hardcore sac ${logdir} 1 default 2000000"
  "bipedal_walker forward_hardcore sac ${logdir} 1 hrs_pot 2000000"
  "bipedal_walker forward_hardcore sac ${logdir} 1 tltl 2000000"
  "bipedal_walker forward_hardcore sac ${logdir} 1 bhnr 2000000"
  "bipedal_walker forward_hardcore sac ${logdir} 1 morl_uni 2000000"
  "bipedal_walker forward_hardcore sac ${logdir} 1 morl_dec 2000000"
  ##
  ## Lunar Lander
  ##
  "lunar_lander land sac ${logdir} 1 default 2000000"
  "lunar_lander land sac ${logdir} 1 hrs_pot 2000000"
  "lunar_lander land sac ${logdir} 1 tltl 2000000"
  "lunar_lander land sac ${logdir} 1 bhnr 2000000"
  "lunar_lander land sac ${logdir} 1 morl_uni 2000000"
  "lunar_lander land sac ${logdir} 1 morl_dec 2000000"
)


DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

for exp in "$@"
do
  index=$(($exp-1))
  n_args=$(echo ${args[$index]} | wc -w)
  #
  if [ $n_args == 7 ]; then
	  $DIR/run_n_exps_docker.sh exp_$exp $(echo ${args[$index]})
  else
	  echo "[$debug_prefix] Invalid exp: exp_$exp"
  fi
done
