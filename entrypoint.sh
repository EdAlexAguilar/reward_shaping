xvfb-run -s "-screen 0 1400x900x24" python run_training.py --env cart_pole_obst --task fixed_height --reward $1 --steps 2000000 --seed 0 --expdir H_95_99
