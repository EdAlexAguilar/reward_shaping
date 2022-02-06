#!/bin/bash

debug=1
debug_steps=100000
short_steps=2000000
long_steps=3000000

if [ $debug -ne 0 ]
then
  short_steps=$debug_steps
  long_steps=$debug_steps
fi


# sequential-run arguments: <env> <task> <algo> <exp_dir> <N-exps> <reward> <steps>
args=(
  ##
  ## Cartpole
  ##
  "cart_pole_obst fixed_height sac ${logdir} 1 default ${short_steps}"
  "cart_pole_obst fixed_height sac ${logdir} 1 hrs_pot ${short_steps}"
  "cart_pole_obst fixed_height sac ${logdir} 1 tltl ${short_steps}"
  "cart_pole_obst fixed_height sac ${logdir} 1 bhnr ${short_steps}"
  "cart_pole_obst fixed_height sac ${logdir} 1 morl_uni ${short_steps}"
  "cart_pole_obst fixed_height sac ${logdir} 1 morl_dec ${short_steps}"
  ##
  ## Bipedal Walker
  ##
  "bipedal_walker forward sac ${logdir} 1 default ${short_steps}"
  "bipedal_walker forward sac ${logdir} 1 hrs_pot ${short_steps}"
  "bipedal_walker forward sac ${logdir} 1 tltl ${short_steps}"
  "bipedal_walker forward sac ${logdir} 1 bhnr ${short_steps}"
  "bipedal_walker forward sac ${logdir} 1 morl_uni ${short_steps}"
  "bipedal_walker forward sac ${logdir} 1 morl_dec ${short_steps}"
  ##
  ## Bipedal Walker Hardcore
  ##
  "bipedal_walker hardcore sac ${logdir} 1 default ${long_steps}"
  "bipedal_walker hardcore sac ${logdir} 1 hrs_pot ${long_steps}"
  "bipedal_walker hardcore sac ${logdir} 1 tltl ${long_steps}"
  "bipedal_walker hardcore sac ${logdir} 1 bhnr ${long_steps}"
  "bipedal_walker hardcore sac ${logdir} 1 morl_uni ${long_steps}"
  "bipedal_walker hardcore sac ${logdir} 1 morl_dec ${long_steps}"
  ##
  ## Lunar Lander
  ##
  "lunar_lander land sac ${logdir} 1 default ${long_steps}"
  "lunar_lander land sac ${logdir} 1 hrs_pot ${long_steps}"
  "lunar_lander land sac ${logdir} 1 tltl ${long_steps}"
  "lunar_lander land sac ${logdir} 1 bhnr ${long_steps}"
  "lunar_lander land sac ${logdir} 1 morl_uni ${long_steps}"
  "lunar_lander land sac ${logdir} 1 morl_dec ${long_steps}"
  ##
  ## F1Tenth
  ##
  "f1tenth informatik ppo ${logdir} 1 default ${short_steps} -novideo"
  "f1tenth informatik ppo ${logdir} 1 min_action ${short_steps} -novideo"
  "f1tenth informatik ppo ${logdir} 1 hrs_pot ${short_steps} -novideo"
  "f1tenth informatik ppo ${logdir} 1 tltl ${short_steps} -novideo"
  "f1tenth informatik ppo ${logdir} 1 bhnr ${short_steps} -novideo"
  "f1tenth informatik ppo ${logdir} 1 morl_uni ${short_steps} -novideo"
  "f1tenth informatik ppo ${logdir} 1 morl_dec ${short_steps} -novideo"
)
