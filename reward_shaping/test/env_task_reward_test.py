import pytest

import numpy as np
from stable_baselines3.common.env_checker import check_env

from reward_shaping.training.utils import make_env, make_agent

env_tasks = {
    "cart_pole_obst": ["fixed_height"],
    "lunar_lander": ["land"],
    "racecar": ["drive_delta", "drive"],
    "bipedal_walker": ["forward", "hardcore"]
}
rewards = ["default", "eval", "tltl", "hprs", "morl_uni", "morl_dec", "bhnr"]



def _generic_env_test(env_name, task, reward_name):
    seed = np.random.randint(0, 1000000)
    env, env_params = make_env(env_name, task, reward_name, eval=True, logdir=None, seed=seed)
    check_env(env)
    for _ in range(1):
        obs = env.reset()
        env.render()
        tot_reward = 0.0
        done = False
        t = 0
        while not done and t < 10:
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            tot_reward += reward
            t += 1
            env.render()
        print(f"[{reward_name}] tot steps: {t}, tot undiscounted reward: {tot_reward:.3f}")
    env.close()
    return True


def _generic_env_test_wt_agent(env_name, model, task, reward_name):
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
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            env.render()
            rr.append(reward)
            tot_reward += reward
            t += 1
        print(f"[{reward_name}] tot steps: {t}, tot reward: {tot_reward:.3f}")
        for k, v in info.items():
            if "counter" in k:
                print(f"{k}: {v}")
    env.close()
    return True


def _generic_training(env, task, reward):
    seed = np.random.randint(0, 1000000)
    train_env, env_params = make_env(env, task, reward, seed=seed)
    eval_env, _ = make_env(env, task, "eval", seed=seed)

    model = make_agent(env, train_env, reward, "sac", logdir=None)
    model.learn(total_timesteps=100, eval_freq=50, eval_env=eval_env)

    train_env.close()
    eval_env.close()
    return True


@pytest.mark.parametrize("env_name, task, reward_name", [(env, task, reward) for env in env_tasks.keys() for task in env_tasks[env] for reward in rewards])
def test_env_task_reward(env_name: str, task: str, reward_name: str):
    assert _generic_env_test(env_name, task, reward_name)


@pytest.mark.parametrize("env_name, task, reward_name", [(env, task, reward) for env in env_tasks.keys() for task in env_tasks[env] for reward in rewards])
def test_train(env_name: str, task: str, reward_name: str):
    assert _generic_training(env_name, task, reward_name)


def _test_eval(self):
    # TODO(Luigi): Rewrite this test using existing checkpoints
    from stable_baselines3 import SAC
    task = "land"
    checkpoint_paths = {
        "default": "/home/luigi/Desktop/logs_iros22/lunar_lander/land_default_sac_Seed320812_1644341625/checkpoint/model_1500000_steps.zip",
        "tltl": "/home/luigi/Desktop/logs_iros22/lunar_lander/land_tltl_sac_Seed143933_1644390812/checkpoint/model_3000000_steps.zip",
        "morl_uni": "/home/luigi/Desktop/logs_iros22/lunar_lander/land_morl_uni_sac_Seed27201_1644350362/checkpoint/model_3000000_steps.zip",
        "morl_dec": "/home/luigi/Desktop/logs_iros22/lunar_lander/land_morl_dec_sac_Seed496777_1644341634/checkpoint/model_3000000_steps.zip",
        "hrs_pos": "/home/luigi/Desktop/logs_iros22/lunar_lander/land_hrs_pot_sac_Seed271147_1644341625/checkpoint/model_1500000_steps.zip"
    }
    for reward, checkpoint_path in checkpoint_paths.items():
        print(f"training reward: {reward}")
        agent = SAC.load(checkpoint_path)
        env_name = None
        _generic_env_test_wt_agent(env_name, agent, task, 'eval')
        print()