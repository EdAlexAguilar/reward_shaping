name='exp1'
env=cart_pole_obst
task=fixed_height
reward=stl
steps=2000000
gpu=all
image=reward_shaping:latest


docker run --rm --name $name --gpus $gpu -d -v $(pwd)/logs:/src/logs --network host $image \
        --env $env --task $task --reward $reward --steps $steps --seed $seed
