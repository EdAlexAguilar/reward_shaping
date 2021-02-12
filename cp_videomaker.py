import vanilla_dqn
import cp_env
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # or any {'0', '1', '2'}
import tensorflow as tf  # to avoid TF import verbosity

def make_video_q_agent(env, agent, target=0.5, save=False, filename='temp', max_steps=200):
    '''
    Best Actions for Q Agent (deterministic policy)
    '''
    # env = gym.wrappers.Monitor(environment, directory="recording", force=True)
    # assert env.env._max_episode_steps >= max_steps
    obs = env.reset()
    env.target = target
    if env.goal is True:
        env.state[4] = env.target
        obs[4] = env.target
    original_env_max_steps = env._max_episode_steps
    if max_steps > env._max_episode_steps:
        env._max_episode_steps = max_steps
    tot_reward = 0
    fig = plt.figure()
    ims = []
    while True:
        obs_tf = tf.convert_to_tensor([obs])
        q_tf = agent(obs_tf)
        action = int(tf.math.argmax(q_tf, axis=1).numpy())
        obs, reward, is_done, _ = env.step(action)
        im = plt.imshow(env.render(mode='rgb_array'), animated=True)
        ims.append([im])
        # test to see if can follow multiple objectives
        # if (obs[0]-env.target) < 0.05:
        #     env.target *= -1
        if is_done:
            break
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

'''
def make_video_prob_agent(env, agent, sampler, save=False, filename='temp', max_steps=500):
    """
    Useful samplers
    np.argmax
    lambda x: np.random.choice(len(x),p=x)
    """
    # env = gym.wrappers.Monitor(environment, directory="recording", force=True)
    if env.env._max_episode_steps < max_steps:
        env.env._max_episode_steps = max_steps
    obs = env.reset()
    env.target = -0.5
    sftmx = tf.nn.softmax
    tot_reward = 0
    steps = 0
    fig = plt.figure()
    ims = []
    while True:
        steps += 1
        obs_tf = tf.convert_to_tensor([obs])
        act_probs_tf = sftmx(agent(obs_tf))
        act_probs = act_probs_tf.numpy()[0]
        action = sampler(act_probs)
        obs, reward, is_done, _ = env.step(action)
        tot_reward += reward
        im = plt.imshow(env.env.render(mode='rgb_array'), animated=True)
        ims.append([im])
        if (obs[0]-env.target) < 0.02:
            env.target *= -1
        if is_done or steps==max_steps:
            break
    if save:
        print("Saving video...")
        if filename=='temp':
            print("Saving to temp.mp4 . Consider renaming!")
        ani = animation.ArtistAnimation(fig, ims, interval=25, blit=True, repeat_delay=300)
        format='.mp4'
        ani.save(filename+format) # Slow process in this manual form
    env.close()
    plt.close()
    return
'''

if __name__ == "__main__":
    env =  vanilla_dqn.Environment()
    # reward_id = 2353.98
    # filename = (f'cartpole_models/target_linrew_deepdouble_dqnmodel_best-{reward_id:.2f}.h5')
    filename = (f'cartpole_models/vanilla_dqn_model_solution.h5')
    agent = tf.keras.models.load_model(filename)
    make_video_q_agent(env, agent, target=0, save=False, filename=filename[:-3], max_steps=200)
