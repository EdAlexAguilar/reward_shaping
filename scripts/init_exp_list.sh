#!/bin/bash

debug=0
debug_steps=1000
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
  #"cart_pole_obst fixed_height sac ${logdir} 1 default ${short_steps} -novideo"
  #"cart_pole_obst fixed_height sac ${logdir} 1 hprs ${short_steps} -novideo"
  #"cart_pole_obst fixed_height sac ${logdir} 1 tltl ${short_steps} -novideo"
  #"cart_pole_obst fixed_height sac ${logdir} 1 bhnr ${short_steps} -novideo"
  #"cart_pole_obst fixed_height sac ${logdir} 1 morl_uni ${short_steps} -novideo"
  #"cart_pole_obst fixed_height sac ${logdir} 1 morl_dec ${short_steps} -novideo"
  ##
  ## Bipedal Walker
  ##
  #"bipedal_walker forward sac ${logdir} 1 default ${short_steps} -novideo"
  #"bipedal_walker forward sac ${logdir} 1 hprs ${short_steps} -novideo"
  #"bipedal_walker forward sac ${logdir} 1 tltl ${short_steps} -novideo"
  #"bipedal_walker forward sac ${logdir} 1 bhnr ${short_steps} -novideo"
  #"bipedal_walker forward sac ${logdir} 1 morl_uni ${short_steps} -novideo"
  #"bipedal_walker forward sac ${logdir} 1 morl_dec ${short_steps} -novideo"
  ##
  ## Bipedal Walker Hardcore
  ##
  #"bipedal_walker hardcore sac ${logdir} 1 default ${long_steps} -novideo"
  #"bipedal_walker hardcore sac ${logdir} 1 hprs ${long_steps} -novideo"
  #"bipedal_walker hardcore sac ${logdir} 1 tltl ${long_steps} -novideo"
  #"bipedal_walker hardcore sac ${logdir} 1 bhnr ${long_steps} -novideo"
  #"bipedal_walker hardcore sac ${logdir} 1 morl_uni ${long_steps} -novideo"
  #"bipedal_walker hardcore sac ${logdir} 1 morl_dec ${long_steps} -novideo"
  ##
  ## Lunar Lander
  ##
  #"lunar_lander land sac ${logdir} 1 default ${long_steps} -novideo"
  #"lunar_lander land sac ${logdir} 1 hprs ${long_steps} -novideo"
  #"lunar_lander land sac ${logdir} 1 tltl ${long_steps} -novideo"
  #"lunar_lander land sac ${logdir} 1 bhnr ${long_steps} -novideo"
  #"lunar_lander land sac ${logdir} 1 morl_uni ${long_steps} -novideo"
  #"lunar_lander land sac ${logdir} 1 morl_dec ${long_steps} -novideo"
  ##
  ## Racecar (Single-agent)
  ##
  #"racecar drive_delta sac ${logdir} 1 default ${short_steps}"
  #"racecar drive_delta sac ${logdir} 1 hprs ${short_steps}"
  #"racecar drive_delta sac ${logdir} 1 tltl ${short_steps}"
  #"racecar drive_delta sac ${logdir} 1 bhnr ${short_steps}"
  #"racecar drive_delta sac ${logdir} 1 morl_uni ${short_steps}"
  #"racecar drive_delta sac ${logdir} 1 morl_dec ${short_steps}"
  ##
  ## Racecar (Multi-agent)
  ##
  "racecar2 follow_delta sac ${logdir} 1 default ${short_steps}"
  "racecar2 follow_delta sac ${logdir} 1 hprs ${short_steps}"
  "racecar2 follow_delta sac ${logdir} 1 tltl ${short_steps}"
  "racecar2 follow_delta sac ${logdir} 1 bhnr ${short_steps}"
  "racecar2 follow_delta sac ${logdir} 1 morl_uni ${short_steps}"
  "racecar2 follow_delta sac ${logdir} 1 morl_dec ${short_steps}"
)
