# Hierarchical Reward Shaping
Reward Shaping Experiments with Temporal Logic for Hierarchical Objectives

To install the dependencies:

```pip install -r requirements.txt```

We assume you run the code from the project directory and that it is included in the `PYTHONPATH`.

### Run training 

To train on cartpole (for all the command line options, see `--help`):

```
python run_training.py --env cart_pole_obst --task fixed_height \ 
                       --reward hrs_pot --steps 2000000 --expdir my_exp
```

This command will start the training for `2M` steps 
using the reward `hrs_pot` (Hierarchical Potential Shaping).
The results will be stored in the directory `logs/cart_pole_obst/my_exp`.


#### Running in headless server
When running on headless server or docker containers, we suggest either to:
- disable the video recording with the flag `-novideo`. e.g.,
```
python run_training.py --env cart_pole_obst --task fixed_height \ 
                       --reward hrs_pot --steps 2000000 --expdir my_exp -novideo
```
- or, to create virtual display with `xvfb`
```
xvfb-run -a -s "-screen 0 1400x900x24" python run_training.py --env cart_pole_obst --task fixed_height \ 
                       --reward hrs_pot --steps 2000000 --expdir my_exp -novideo 
```

#### Run training via Docker

To train via Docker:
- pull the image from Dockerhub: `docker pull luigiberducci/reward_shaping`
- or, build the image: `docker build -t reward_shaping .`

Then start the training:

```
docker run --name exp_cpole --rm -it \
	       -u $(id -u):$(id -g) -v $(pwd):/src \
	       --gpus all <image-name> \
	       /bin/bash entrypoint.sh my_exp cart_pole_obst fixed_height hrs_pot 2000000 1 -no_video
```

This command will start the training for 2M steps using the reward `hrs_pot` (STL-Hierarchical).
The results will be stored in the directory `logs/cart_pole_obst/my_exp`.

#### Run training with SLURM and Singularity

If working on a scientific cluster, you can perform the following steps:
1. Build the singularity image from the docker hub: `singularity build reward_shaping.sif docker://luigiberducci/reward_shaping:latest`
1. Run a slurm script, eg., see `scripts/start_slurm_exps.sh`
