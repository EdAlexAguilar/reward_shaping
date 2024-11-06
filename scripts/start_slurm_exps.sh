#!/bin/sh

#SBATCH -J array
#SBATCH -p vsc3plus_0064
#SBATCH --qos vsc3plus_0064
#SBATCH -N 1
#SBATCH --ntasks-per-node 5 
#SBATCH --ntasks-per-core 1 
#SBATCH --cpus-per-task 4   
#SBATCH --array=1,7,13,19
#SBATCH --output=slurm-%j-%A-%a.out
#SBATCH --error=slurm-%j-%A-%a.err

module purge
module load go/1.13.15 singularity/3.8.3

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
image="${DIR}/reward_shaping.sif"

# parse args
source ${DIR}/scripts/init_exp_list.sh && echo "Loaded ${#args[@]} param configurations"

aa=${args[$(($SLURM_ARRAY_TASK_ID % ${#args[@]}))]}
expdir="exps_$(date '+%d%m%Y')"
env=$(echo $aa | cut -d ' ' -f 1)
task=$(echo $aa | cut -d ' ' -f 2)
algo=$(echo $aa | cut -d ' ' -f 3)
n_seeds=$(echo $aa | cut -d ' ' -f 4)
reward=$(echo $aa | cut -d ' ' -f 5)
steps=$(echo $aa | cut -d ' ' -f 6)
novideo=$(echo $aa | cut -d ' ' -f 7)

# pre
echo "Configuration:"
echo -e "\t datetime: $(date "+%d-%m-%Y, %H:%M:%S")"
echo -e "\t dir: ${DIR}"
echo -e "\t Singularity image: ${image}"
echo -e "\t args: ${aa}"
echo ""

# run
start_time=$(date +%s)

singularity exec $image /bin/bash entrypoint.sh $expdir $env $task $algo $reward $steps $n_seeds $novideo > /dev/null

status=$?
end_time=$(date +%s)

# post
elapsed_time=$(( end_time - start_time ))
echo "Done. Informations:"
echo -e "\t datetime: $(date "+%d-%m-%Y, %H:%M:%S")"
echo -e "\t elapsed time (seconds): ${elapsed_time}"
echo -e "\t exit status: ${status}"

