"""
Cartpole Experiments w/TF2
"""
import gym
import collections
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # or any {'0', '1', '2'}
import tensorflow as tf  # to avoid TF import verbosity
import cp_videomaker as video
import cp_env


# CONSTANTS
BATCH_SIZE = 64
GAMMA = 0.98
BUFFER_SIZE = int(2e4)
SYNC_NETWORKS = 2e3
MIN_BUFFER_TRAINING = 5e3
INIT_LR = 5e-4
EPSILON_MIN = 0.02
EPSILON_DECAY_STEPS = 1e4


class Environment(cp_env.CartPoleEnv):
    """
    State of Environment:
    if goal is False
        x, x_dot, theta, theta_dot
    elif goal is True:
        x, x_dot, theta, theta_dot, target

    End Conditions
    x_threshold: if abs(x)>threshold
    theta_threshold: if abs(theta)>threshold
    max_steps: if self.step_count > max_steps

    Use this class to override the default reward function.
    """
    def __init__(self, x_threshold=2.5, theta_threshold_deg=24, max_steps=200, goal=False):
        super().__init__(x_threshold, theta_threshold_deg, max_steps, goal)
        self.var_theta = 10
        self.var_dist = 0.5
        self.safe_angle_tolerance = 30*math.pi/360

    def linear_decay(self, z, z0, scale, max_val=1, min_val=0):
        return max(min_val, max_val - abs((z-z0)/scale))

    '''
    # Vanilla Reward implemented in super() class.
    def reward(self):
        if abs(self.state[2]) < self.theta_threshold_radians:
            return 1.0
        else:
            return 0.0

    # Example Reward Modification
    def reward(self):
        """Be careful with dimension of self.state
        if goal=True: state includes target
        else: state = x, x_dot, theta, theta_dot"""
        if self.goal:
            x, x_dot, theta, theta_dot, target = self.state
        else:
            x, x_dot, theta, theta_dot = self.state
        r0 = self.linear_decay(theta, 0, self.safe_angle_tolerance)
        id0 = 1 if (abs(theta) < self.safe_angle_tolerance) else 0
        r1 = self.linear_decay(x, target, 1)
        return r0  + id0*r1

    '''


def DQN(goal=False):
    state_dim = 5 if goal else 4
    kern_init=tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_in',
                                                    distribution='truncated_normal')
    model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(32, input_shape=(state_dim,), activation='relu', kernel_initializer=kern_init),
    tf.keras.layers.Dense(32, activation='relu', kernel_initializer=kern_init),
    tf.keras.layers.Dense(2, kernel_initializer=kern_init)
    ])
    loss_fn = tf.keras.losses.MSE
    model.compile(optimizer='adam', loss=loss_fn)
    return model

EXP = collections.namedtuple('Experience',
          field_names=['state', 'action', 'reward', 'state_1', 'done'])
# Convention: variable_i is variable i steps later
# variable_0 = variable

class MemoryBuffer:
    def __init__(self, maxsize):
        self.buffer = collections.deque(maxlen=maxsize)

    def __len__(self):
        return len(self.buffer)

    def append(self, exp):
        self.buffer.append(exp)

    def sample(self, batch_size, n=1):
        rand_indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        state, action, reward, state_1, done = zip(*[self.buffer[i] for i in rand_indices])
        return np.array(state), np.array(action), np.array(reward), np.array(state_1), \
               np.array(done, dtype=np.uint8)

class Agent:
    def __init__(self, environment, experience_buffer):
        self.env = environment
        self.exp_buffer = experience_buffer
        self._reset()

    def _reset(self):
        self.state = self.env.reset()
        self.total_reward = 0.0

    def step(self, model, epsilon=0.0):
        '''
        epsilon-greedy step , eps=0 is argmax deterministic
        '''
        episode_reward = None
        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            # state = np.array([self.state], copy=False)
            state_tf = tf.convert_to_tensor([self.state])
            Q_tf =  tf.stop_gradient(model(state_tf))
            action = int(tf.argmax(Q_tf, axis=1))
        state_1, reward, done, _ = self.env.step(action)
        self.total_reward += reward
        exp = EXP(self.state, action, reward, state_1, done)
        self.exp_buffer.append(exp)
        self.state = state_1
        if done:
            episode_reward = self.total_reward
            self._reset()
        return episode_reward

def epsilon(step, min_eps=EPSILON_MIN, decay_steps=EPSILON_DECAY_STEPS):
    if step<decay_steps:
        return 1 - step*(1-min_eps)/decay_steps
    else:
        return min_eps

def calculate_loss(model, ref_model, batch, gamma=GAMMA):
    state, action, reward, state_1, done = batch
    targets = gamma * tf.math.reduce_max(ref_model(state_1), axis=1)
    targets = tf.multiply(targets, (done+1)%2) + reward
    q_vals_sa = tf.gather(model(state), action, batch_dims=1)
    loss = tf.keras.losses.MSE(tf.stop_gradient(targets), q_vals_sa)
    return loss

if __name__ == "__main__":
    # Vanilla version does not have a goal
    # If set to true, modify the reward function of the environment
    goal = False
    env = Environment(goal=goal)
    dqn = DQN(goal=goal)
    ref_dqn = DQN(goal=goal) # Also referred to as a "target network"
    buffer = MemoryBuffer(BUFFER_SIZE)
    agent = Agent(env, buffer)
    optimizer = tf.keras.optimizers.Adam(learning_rate=INIT_LR)

    SOLVED_REWARD_BOUND = int(0.9*env._max_episode_steps)

    step_num = 0
    step_timing = 0
    timing = time.time()
    tot_time = timing
    tot_rewards = []
    best_reward = None
    # Exploration & Training Loop
    while True:
        step_num += 1
        eps = epsilon(step_num)
        episode_reward = agent.step(dqn, eps)
        if episode_reward is not None:
            tot_rewards.append(episode_reward)
            speed = (step_num - step_timing)/(time.time() - timing + 1e-8)
            step_timing = step_num
            timing = time.time()
            running_mean_reward = np.mean(tot_rewards[-25:])
            print(f'{step_num} Steps taken :: {len(tot_rewards)} Episodes ::'
                  f' {eps:.3f} Epsilon :: {speed:.1f} Steps/Sec ::'
                  f' {running_mean_reward:.2f} Running Mean')
            if best_reward is None or best_reward < running_mean_reward:
                # if you want to save intermediate models ; but since this is a small problem not necessary
                # dqn.save(f'cartpole_models/vanilla_dqn_model_best-{running_mean_reward:.2f}.h5')
                if best_reward is not None:
                    print(f'Best Reward Update: {best_reward:.2f}->{running_mean_reward:.2f}')
                    # video.make_video_q_agent(env, dqn, max_steps=250)
                best_reward = running_mean_reward
            if best_reward > SOLVED_REWARD_BOUND:
                print(f'Solved in {time.time()-tot_time:.1f}Sec!')
                dqn.save(f'cartpole_models/vanilla_dqn_model_solution.h5')
                break
        if len(buffer) < MIN_BUFFER_TRAINING:
            continue
        if step_num % SYNC_NETWORKS == 0:
            ref_dqn.set_weights(dqn.get_weights())
        batch = buffer.sample(BATCH_SIZE)
        with tf.GradientTape() as tape:
            loss = calculate_loss(dqn, ref_dqn, batch)
        grads = tape.gradient(loss, dqn.trainable_variables)
        optimizer.apply_gradients(zip(grads, dqn.trainable_variables))
