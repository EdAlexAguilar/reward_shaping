#!/bin/sh
#SBATCH -J array
#SBATCH -N 1
#SBATCH --array=1-31
#SBATCH --output=../logs/slurm-%x.%j.out
#SBATCH --error=../logs/slurm-%x.%j.err

source ./init_exp_list.sh   # load list of args

echo "args: "  $args[$SLURM_ARRAY_TASK_ID]