#!/bin/sh

#SBATCH -J array
#SBATCH -N 1
#SBATCH --array=1-6
#SBATCH --output=slurm-%j-%A-%a.out
#SBATCH --error=slurm-%j-%A-%a.err

module purge
module load go/1.13.15 singularity/3.8.3

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
image="${DIR}/reward_shaping.sif"

# parse args
source ${DIR}/scripts/init_exp_list.sh && echo "Loaded ${#args[@]} param configurations"

aa=${args[$SLURM_ARRAY_TASK_ID]}
expdir="exps_$(date '+%d%m%Y')"
env=$(echo $aa | cut -d ' ' -f 1)
task=$(echo $aa | cut -d ' ' -f 2)
algo=$(echo $aa | cut -d ' ' -f 3)
n_seeds=$(echo $aa | cut -d ' ' -f 4)
reward=$(echo $aa | cut -d ' ' -f 5)
steps=$(echo $aa | cut -d ' ' -f 6)
novideo=$(echo $aa | cut -d ' ' -f 7)

# run
echo "Dir: ${DIR}"
echo "Image SIF: ${image}"
echo "ARGS: ${aa}"

singularity exec $image /bin/bash entrypoint.sh $expdir $env $task $algo $reward $steps $n_seeds $novideo > /dev/null

echo "done with exit status $?"


