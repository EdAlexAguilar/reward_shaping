import numpy as np
from stable_baselines3.common.env_checker import check_env

from reward_shaping.training.utils import make_env, make_agent


def generic_env_test(env_name, task, reward_name):
    seed = np.random.randint(0, 1000000)
    env, env_params = make_env(env_name, task, reward_name, eval=True, logdir=None, seed=seed)
    # check
    check_env(env)
    # evaluation
    for _ in range(1):
        obs = env.reset()
        env.render()
        tot_reward = 0.0
        done = False
        t = 0
        while not done:
            t += 1
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            print(reward)
            tot_reward += reward
            env.render()
        print(f"[{reward_name}] tot steps: {t}, tot reward: {tot_reward:.3f}")
    env.close()
    return True


def generic_env_test_wt_agent(env_name, model, task, reward_name):
    seed = np.random.randint(0, 1000000)
    env, env_params = make_env(env_name, task, reward_name, eval=True, logdir=None, seed=seed)
    # check
    check_env(env)
    # evaluation
    for _ in range(1):
        obs = env.reset()
        env.render()
        tot_reward = 0.0
        done = False
        t = 0
        rr = []
        speeds = []
        steers = []
        frames = []
        while not done:
            t += 1
            if t % 100 == 0:
                print(t)
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            speeds.append(obs["speed"])
            steers.append(obs["steering"])
            rr.append(reward)
            tot_reward += reward
            frames.append(env.render(mode="follow"))
        print(f"[{reward_name}] tot steps: {t}, tot reward: {tot_reward:.3f}")
        for k, v in info.items():
            if "counter" in k:
                print(f"{k}: {v}")
        import matplotlib.pyplot as plt
        plt.plot(speeds, label="speed cmd")
        plt.plot(steers, label="steer cmd")
        plt.legend()
        plt.show()
        # make video
        import imageio
        writer = imageio.get_writer(f'video_racecar.mp4', fps=100 // 5)
        for image in frames:
            writer.append_data(image)
        writer.close()
    return True


def generic_training(env, task, reward):
    # create training environment
    seed = np.random.randint(0, 1000000)
    train_env, env_params = make_env(env, task, reward, seed=seed)
    eval_env, _ = make_env(env, task, "eval", seed=seed)
    # create agent
    model = make_agent(env, train_env, reward, "sac", logdir=None)
    # train
    model.learn(total_timesteps=1000, eval_freq=500, eval_env=eval_env)
    train_env.close()
