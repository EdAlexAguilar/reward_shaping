# Hierarchical Reward Shaping
Reward Shaping Experiments with Temporal Logic for Hierarchical Objectives

To install the dependencies:

```pip install -r requirements.txt```

### Run training 

To train on cartpole (for all the command line options, see `--help`):

```
python run_training.py --env cart_pole_obst --task fixed_height \ 
                       --reward gb_cr_bi --steps 2000000 --expdir my_exp
```

This command will start the training for 2M steps using the reward `gb_bcr_bi` (graph-based with binary indicators).
The results will be stored in the directory `logs/my_exp`.

### Run training via Docker

To train via Docker, first build the image:

```docker build -t reward_shaping .```

Then start the training:

```
docker run --name exp_cartpole_gbased --rm -it \
	       -u $(id -u):$(id -g) -v $(pwd):/src \
	       --gpus all reward_shaping \
	       /bin/bash entrypoint.sh cart_pole_obst fixed_height my_exp gb_bcr_bi
```

This command will start the training for 2M steps using the reward `gb_bcr_bi` (graph-based with binary indicators).
The results will be stored in the directory `logs/my_exp`.


