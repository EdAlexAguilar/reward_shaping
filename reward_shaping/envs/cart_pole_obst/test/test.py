from stable_baselines3.common.env_checker import check_env

from reward_shaping.training.utils import make_env


def generic_env_test(env_name, task, reward_name, potential=False):
    env, env_params = make_env(env_name, task, reward_name, use_potential=potential, eval=True, logdir=None, seed=0)
    # check
    check_env(env)
    # evaluation
    _ = env.reset()
    env.render()
    tot_reward = 0.0
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print(reward)
        tot_reward += reward
        env.render()
    print(f"[{reward_name}] tot reward: {tot_reward:.3f}")
    env.close()
    return True