#!/bin/bash

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
  "bipedal_walker hardcore sac ${logdir} 1 default 2000000"
  "bipedal_walker hardcore sac ${logdir} 1 hrs_pot 2000000"
  "bipedal_walker hardcore sac ${logdir} 1 tltl 2000000"
  "bipedal_walker hardcore sac ${logdir} 1 bhnr 2000000"
  "bipedal_walker hardcore sac ${logdir} 1 morl_uni 2000000"
  "bipedal_walker hardcore sac ${logdir} 1 morl_dec 2000000"
  ##
  ## Lunar Lander
  ##
  "lunar_lander land sac ${logdir} 1 default 2000000"
  "lunar_lander land sac ${logdir} 1 hrs_pot 2000000"
  "lunar_lander land sac ${logdir} 1 tltl 2000000"
  "lunar_lander land sac ${logdir} 1 bhnr 2000000"
  "lunar_lander land sac ${logdir} 1 morl_uni 2000000"
  "lunar_lander land sac ${logdir} 1 morl_dec 2000000"
  ##
  ## F1Tenth
  ##
  "f1tenth informatik ppo ${logdir} 1 default 1000000 -novideo"
  "f1tenth informatik ppo ${logdir} 1 min_action 1000000 -novideo"
  "f1tenth informatik ppo ${logdir} 1 hrs_pot 1000000 -novideo"
  "f1tenth informatik ppo ${logdir} 1 tltl 1000000 -novideo"
  "f1tenth informatik ppo ${logdir} 1 bhnr 1000000 -novideo"
  "f1tenth informatik ppo ${logdir} 1 morl_uni 1000000 -novideo"
  "f1tenth informatik ppo ${logdir} 1 morl_dec 1000000 -novideo"
)
