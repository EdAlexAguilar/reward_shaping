#!/bin/sh

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

#SBATCH -J array
#SBATCH -N 1
#SBATCH --array=1-6
#SBATCH --output=${dir}/logs/slurm-%j-%A-%a.out
#SBATCH --error=${dir}/logs/slurm-%j-%A-%a.err

module purge
module load singularity

source ./init_exp_list.sh   # load list of args

aa=${args[$SLURM_ARRAY_TASK_ID]}
image="../reward_shaping.sif"

echo $aa