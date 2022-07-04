import pathlib

from reward_shaping.training.utils import make_env

logdir = pathlib.Path("logs/tmp")
logdir.mkdir(exist_ok=True, parents=True)
seed = 0

train_env, trainenv_params = make_env("racecar", "drive", "default", eval=True, logdir=logdir, seed=seed)

train_env.reset()
done = False

while not done:
    action = train_env.action_space.sample()
    obs, reward, done, info = train_env.step(action)
    print(reward)

train_env.close()
print("done")

