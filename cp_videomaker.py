import cp_continuousobstacle_env
import ppo_mp as ppo
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # or any {'0', '1', '2'}
import tensorflow as tf  # to avoid TF import verbosity
import numpy as np

def make_video_ppo_agent(env, agent, x_init=None, save=False, filename='temp', max_steps=200):
    '''
    Best Actions for Continuous Agent (deterministic policy, action = mu)
    '''
    obs = env.reset()
    if x_init is not None:
        env.state[0] = x_init
        obs[0] = x_init
    original_env_max_steps = env._max_episode_steps
    if max_steps > env._max_episode_steps:
        env._max_episode_steps = max_steps
    tot_reward = 0
    fig = plt.figure()
    ims = []
    while True:
        obs_tf = tf.convert_to_tensor([obs])
        mu, var = tf.stop_gradient(agent(obs_tf))
        action = np.clip(mu.numpy()[0], -1, 1) # No exploration, use mu directly
        obs, reward, is_done, _ = env.step(action)
        '''
        if abs(obs[0])<0.05:
            x, x_d, th, th_d, b = env.state
            env.state = (x_init, x_d, th, th_d, b)
            obs[0] = x_init
        '''
        image = env.render(mode='rgb_array')
        im = plt.imshow(image, animated=True)
        ims.append([im])
        tot_reward += reward
        if is_done:
            print(f'Battery Level: {obs[4]:.2f}%')
            break
    print(f'Total Reward: {tot_reward:.2f}')
    if save:
        print("Saving video...")
        if filename=='temp':
            print("Saving to temp.mp4 . Consider renaming!")
        ani = animation.ArtistAnimation(fig, ims, interval=25, blit=True, repeat_delay=300)
        format='.mp4'
        ani.save(filename+format) # Saving is a slow process
    env._max_episode_steps = original_env_max_steps
    env.close()
    plt.close()
    return



if __name__ == "__main__":
    MAX_STEPS = ppo.MAX_STEPS
    X_THRESHOLD = ppo.X_THRESHOLD
    THETA_THRESHOLD = ppo.THETA_THRESHOLD
    num_agents = int(ppo.CORES)

    # NUM_LEVELS = 6
    # obstacle_widths = np.linspace(0.01, 0.08, num=NUM_LEVELS)
    # obstacle_heights = np.linspace(0.005, 0.1, num=NUM_LEVELS)
    obstacle_widths = [0.08]
    obstacle_heights = [0.1]

    names = ['weighted_05', 'indicator_narrow_1']
    params = {names[0]: np.array([0.575, 0.285, 0.14]),
              names[1]: np.array([ppo.REWARD_THETA_BOUND, ppo.REWARD_X_BOUND])}
    envs = {names[0]: ppo.EnvironmentWeightedReward(params[names[0]]),
            names[1]: ppo.EnvironmentIndicatorReward(params[names[1]])}

    actor = ppo.Actor()

    for i in range(num_agents):
        reward_type = 1 #
        level = 0
        env = envs[names[reward_type]]
        env.set_obstacle_width_height(obstacle_widths[level], obstacle_heights[level])

        load_path= f'{names[reward_type]}_L{level}_{i}'

        actor.load_weights(f'models\ppo_actor_{load_path}.ckpt')
        make_video_ppo_agent(env, actor, x_init=None, save=False, filename=f'{names[reward_type]}_L{level}', max_steps=MAX_STEPS)
