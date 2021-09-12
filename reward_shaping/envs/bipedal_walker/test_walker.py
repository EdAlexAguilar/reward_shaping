from stable_baselines3.common.env_checker import check_env

from reward_shaping.training.utils import make_env


def main(reward):
    env_name = "bipedal_walker"
    task = "forward"
    env, env_params = make_env(env_name, task, reward)

    # evaluation
    obs = env.reset()
    env.render()
    rewards = []
    tot_reward = 0.0
    for i in range(10000):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        tot_reward += reward
        env.render()
        if done:
            rewards.append(tot_reward)
            obs = env.reset()
            print(f"reward: {tot_reward:.3f}")
            tot_reward = 0.0
            input()
    try:
        check_env(env)
        result = True
    except Exception as err:
        result = False
        print(err)
    print(f"Check env: {result}")


if __name__ == "__main__":
    main("gb_bpr_ci")
