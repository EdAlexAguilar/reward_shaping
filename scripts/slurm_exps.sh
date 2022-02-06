#!/bin/sh

#SBATCH -J array
#SBATCH -N 1
#SBATCH --array=1-6

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

#SBATCH --output=${DIR}/../logs/slurm-%j-%A-%a.out
#SBATCH --error=${DIR}/../logs/slurm-%j-%A-%a.err

module purge
module load singularity

source ./init_exp_list.sh   # load list of args

aa=${args[$SLURM_ARRAY_TASK_ID]}
image="${DIR}/../reward_shaping.sif"

echo $aa