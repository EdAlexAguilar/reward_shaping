import cp_continuous_env
import cp_ppo_utils
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
        print(f'Battery Level: {obs[4]:.2f}%')
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
    MAX_STEPS = 500
    X_THRESHOLD = 2.5
    THETA_THRESHOLD = 24
    env = cp_continuous_env.CartPoleContEnv(x_threshold=X_THRESHOLD, theta_threshold_deg=THETA_THRESHOLD,
                                max_steps=MAX_STEPS)
    env.reset()
    filename = (f'ppo_actor_weighted_05_solution.tf')
    actor = cp_ppo_utils.Actor()
    actor = tf.keras.models.load_model(filename)
    make_video_ppo_agent(env, actor, x_init=None, save=False, filename=filename[:-3], max_steps=MAX_STEPS)
