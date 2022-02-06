#!/bin/sh
#SBATCH -J array
#SBATCH -N 1
#SBATCH --array=1-31
#SBATCH --output=../logs/slurm-%j-%A-%a.out
#SBATCH --error=../logs/slurm-%j-%A-%a.err

source ./init_exp_list.sh   # load list of args

echo "args: "  $args[$SLURM_ARRAY_TASK_ID]
echo ""