# reward_shaping
Reward Shaping Experiments with Temporal Logic for Hierarchical Objectives 

Packages Needed (I just installed the latest for most)
python 3.8.0 (or higher)

gym==0.18.0

tensorflow==2.3.0

h5py==2.10.0

matplotlib==3.3.4

numpy==1.19.2



For a simple demo:

run vanilla_dqn.py

this will train to vanilla Q-Learning on vanilla cartpole and save the solution (it took 140s on my laptop)
then you can run

cp_videomaker.py

to see the agent in action (by default it will show the video, and not save it)

I modified the environment class, so we need to use the cp_env.py module and not the gym cartpole environment.
The modifications are just to be able to incorporate the 'goal' state, and some minor graphical changes to the rendering.

If you want to play further:

you can use the vanilla_dqn as a template
in the __main__ part, set goal=True
and at the top of the file there is the Environment Class
Here you can override the default reward (I show an example) and see how it performs.

