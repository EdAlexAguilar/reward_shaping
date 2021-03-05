import gym
import collections
import numpy as np
import math
import matplotlib.pyplot as plt
import os
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # or any {'0', '1', '2'}
import tensorflow as tf  # to avoid TF import verbosity
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model
import cp_continuous_env
import cp_ppo_videomaker

# Training Constants
# ------------------
GAMMA = 0.99
GAE_LAMBDA = 0.95 # Generalized Advantage Estimate Parameter

TRAJECTORY_SIZE = 2049
LEARNING_RATE_ACTOR = 1e-4
LEARNING_RATE_CRITIC = 1e-3
ENTROPY_BETA = 1e-4 # To promote exploration
PPO_EPS = 0.25 # Clipping Value
PPO_EPOCHES = 10
PPO_BATCH_SIZE = 64
assert (TRAJECTORY_SIZE-1)%PPO_BATCH_SIZE == 0

TEST_ITERS = 10 # After how many trajectories checks progress
# Episodic Constants
MAX_STEPS = 200
SOLVED_REWARD_BOUND = int(0.75*MAX_STEPS)



def logprob_gauss(mu, sigma, action, eps=1e-4):
    """
    Meant to handle tensors
    Returns Log(Gauss_{mu,sigma}(action))
    """
    lpg_1 = - ((mu - action) ** 2) / (2*sigma**2 + eps)
    lpg_2 = - 0.5*tf.math.log(2 * np.pi * sigma**2)
    return lpg_1 + lpg_2

def Critic():
    state_dim = 5
    kern_init=tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_in',
                                                    distribution='truncated_normal')
    model = tf.keras.models.Sequential([
    Dense(32, input_shape=(state_dim,), activation='relu', kernel_initializer=kern_init),
    Dense(32, activation='relu', kernel_initializer=kern_init),
    Dense(1, kernel_initializer=kern_init)
    ])
    loss_fn = tf.keras.losses.MSE
    model.compile(optimizer='adam', loss=loss_fn)
    return model

class Actor(Model):
    def __init__(self):
        super(Actor, self).__init__()
        self.state_dim = 5
        self.kern_init = tf.keras.initializers.VarianceScaling(scale=1.0, mode='fan_in',
                                                    distribution='truncated_normal')
        self.dense1 = Dense(32, input_shape=(self.state_dim,), activation='relu',
                            kernel_initializer=self.kern_init)
        self.dense2 = Dense(32, activation='relu', kernel_initializer=self.kern_init)
        self.mu = Dense(1, activation='tanh')
        self.var = Dense(1, activation='softplus')

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        mu = self.mu(x)
        var = self.var(x)
        return mu, var

EXP = collections.namedtuple('Experience',
          field_names=['state', 'action', 'reward', 'state_1', 'done'])

def test_policy(actor, environment, count=10):
    rewards = 0.0
    steps = 0
    for _ in range(count):
        state = environment.reset()
        while True:
            state_tf = tf.convert_to_tensor([state])
            mu, var = tf.stop_gradient(actor(state_tf))
            action = np.clip(mu.numpy()[0], -1, 1) # No exploration, use mu directly
            state, reward, done, _ = environment.step(action)
            rewards += reward
            steps += 1
            if done:
                break
    return rewards/count, steps/count

def play_trajectory(actor, environment, traj_size=TRAJECTORY_SIZE):
    """
    Assumes environment has been initialized previously
    Clips Random actions to [-1, 1]
    """
    trajectory = []
    state = environment.state
    for _ in range(traj_size):
        state_tf = tf.convert_to_tensor([state])
        mu, var = tf.stop_gradient(actor(state_tf))
        sigma = np.sqrt(var.numpy()[0])
        mu = mu.numpy()[0]
        action = np.clip(np.random.normal(mu, sigma), -1, 1)
        state_1, reward, done, info = environment.step(action)
        exp = EXP(state, action, reward, state_1, done)
        trajectory.append(exp)
        state = state_1
        if done:
            state = environment.reset()
    return trajectory

def advantage_referencereward(trajectory, critic, gamma=GAMMA, lam=GAE_LAMBDA):
    """
    Assumes trajectory list is filled with EXP tuples
    delta = r_t - V_t + gamma*V_{t+1}
    """
    states_tf = [tf.convert_to_tensor([exp.state]) for exp in trajectory]
    values = [tf.stop_gradient(critic(state)).numpy()[0] for state in states_tf]
    gae = 0.0
    advantages = []
    val_references = []
    for val, val_1, exp in zip(reversed(values[:-1]), reversed(values[1:]), reversed(trajectory[:-1])):
        if exp.done:
            delta = exp.reward - val
            gae = delta
        else:
            delta = exp.reward + gamma * val_1 - val
            gae = delta + gamma * lam * gae
        advantages.append(gae)
        val_references.append(gae + val)
    return list(reversed(advantages)), list(reversed(val_references))


def train_PPO(env, name='vanilla', solved_bound=SOLVED_REWARD_BOUND):
    actor = Actor()
    critic = Critic()
    env.reset()
    optimizer_actor = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_ACTOR)
    optimizer_critic = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_CRITIC)

    step_num = 0 # Counts Number of trajectories sampled
    step_timing = 0
    timing = time.time()
    tot_time = timing
    reward_test = [test_policy(actor, env)]
    print(f'----------------- INITIALIZING -----------------')
    print(f'Initial Avg Reward {reward_test[-1][0]:.2f}  :: Initial Avg Steps {reward_test[-1][1]:.2f}')
    best_reward = reward_test[-1][0]
    while True:
        step_num += 1
        if step_num % TEST_ITERS == 0:
            reward_test.append(test_policy(actor, env))
            time_diff = time.time() - timing
            timing = time.time()
            print(f'{step_num} Trajectories :: Time per Trajectory {time_diff/TEST_ITERS:.1f} Sec ::'
                  f' {reward_test[-1][0]:.2f} Avg Rewards ::  {reward_test[-1][1]:.2f} Avg Steps')
            if best_reward < reward_test[-1][0]:
                best_reward = reward_test[-1][0]
                # actor.save(f'ppo_actor_{name}_{best_reward:.2f}.tf', save_format="tf")
                # critic.save(f'ppo_critic_{name}_{best_reward:.2f}.h5')
            if best_reward > solved_bound:
                print(f'Solved in {time.time()-tot_time:.1f} Sec!')
                actor.save(f'ppo_actor_{name}_solution.tf', save_format="tf")
                critic.save(f'ppo_critic_{name}_solution.h5')
                cp_ppo_videomaker.make_video_ppo_agent(env, actor)
                return reward_test


        trajectory = play_trajectory(actor, env)
        advantages, references = advantage_referencereward(trajectory, critic)
        advantages_tf = tf.convert_to_tensor(advantages)
        references_tf = tf.convert_to_tensor(references)

        traj_states_tf = [tf.convert_to_tensor([exp.state], dtype=tf.float32) for exp in trajectory]
        traj_actions_tf = [tf.convert_to_tensor([exp.action], dtype=tf.float32) for exp in trajectory]
        mu_tf, var_tf = tf.stop_gradient(actor(tf.convert_to_tensor(traj_states_tf)))
        sigma_tf = tf.sqrt(var_tf)

        old_logprob_tf = tf.stop_gradient(logprob_gauss(mu_tf, sigma_tf, traj_actions_tf))

        # normalize advantages
        advantages_tf = (advantages-tf.math.reduce_mean(advantages))/tf.math.reduce_std(advantages)

        # drop last entry from the trajectory, an our adv and ref value calculated without it
        trajectory = trajectory[:-1]
        old_logprob_tf = tf.reshape(old_logprob_tf[:-1], [len(trajectory), 1])

        policy_loss_report = 0.0
        value_loss_report = 0.0
        for epoch in range(PPO_EPOCHES):
            for batch in range(0, len(trajectory), PPO_BATCH_SIZE):
                batch_states_tf = tf.concat(traj_states_tf[batch:batch + PPO_BATCH_SIZE], axis=0)
                batch_actions_tf = tf.concat(traj_actions_tf[batch:batch + PPO_BATCH_SIZE], axis=0)
                batch_advantages_tf = advantages_tf[batch:batch + PPO_BATCH_SIZE]
                batch_references_tf = references_tf[batch:batch + PPO_BATCH_SIZE]
                batch_old_logprob_tf = old_logprob_tf[batch:batch + PPO_BATCH_SIZE]

                with tf.GradientTape() as tape_critic:
                    batch_values_tf = critic(batch_states_tf)
                    loss_value = tf.reduce_mean(tf.keras.losses.MSE(batch_values_tf, batch_references_tf))

                grads_critic = tape_critic.gradient(loss_value, critic.trainable_variables)
                optimizer_critic.apply_gradients(zip(grads_critic, critic.trainable_variables))
                value_loss_report += loss_value.numpy()
                # breakpoint()

                with tf.GradientTape() as tape_actor:
                    batch_mus_tf, batch_vars_tf = actor(batch_states_tf)
                    batch_sigmas_tf = tf.sqrt(batch_vars_tf)
                    batch_logprob_tf = logprob_gauss(batch_mus_tf, batch_sigmas_tf, batch_actions_tf)
                    ratio_policies = tf.math.exp(batch_logprob_tf - batch_old_logprob_tf)
                    ppo_objective_tf = batch_advantages_tf * ratio_policies
                    clipped_ratio = tf.clip_by_value(ratio_policies, 1.0 - PPO_EPS, 1.0 + PPO_EPS)
                    ppo_objective_clipped_tf = batch_advantages_tf * clipped_ratio
                    loss_policy = - tf.math.reduce_mean(tf.math.minimum(ppo_objective_tf, ppo_objective_clipped_tf))
                    entropy_tf = - (tf.math.log(2*np.pi*batch_vars_tf)+1)/2
                    entropy_loss = ENTROPY_BETA*tf.math.reduce_mean(entropy_tf)
                    loss_policy += entropy_loss

                grads_actor = tape_actor.gradient(loss_policy, actor.trainable_variables)
                optimizer_actor.apply_gradients(zip(grads_actor, actor.trainable_variables))
                policy_loss_report += loss_policy.numpy()
        print(f'    Value Loss: {value_loss_report:.2f}  :: Policy Loss: {policy_loss_report:.3f}')
