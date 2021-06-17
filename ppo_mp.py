import collections
import numpy as np
import os
import time
import multiprocessing as mp
# import concurrent.futures  # Need to look into this
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}
import tensorflow as tf  # to avoid TF import verbosity
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model
from envs.cart_pole import cp_continuousobstacle_env

# Set TF to work on CPU
tf.config.set_visible_devices([], 'GPU')
# Number of Cores available
CORES = mp.cpu_count()


# Training Constants
# ------------------
GAMMA = 0.99
GAE_LAMBDA = 0.9 # Generalized Advantage Estimate Parameter

TRAJECTORY_SIZE = 8193
LEARNING_RATE_ACTOR = 1e-5  # 3e-4
LEARNING_RATE_CRITIC = 1e-5  # 3e-4
ENTROPY_BETA = 8e-3 # 5e-4 # To promote exploration
PPO_EPS = 0.25 # Clipping Value
PPO_EPOCHES = 12
PPO_BATCH_SIZE = 256
assert (TRAJECTORY_SIZE-1)%PPO_BATCH_SIZE == 0

TEST_ITERS = 4 # After how many trajectories checks progress
MAX_TRAJECTORIES = 720 # how many trajectories before stopping training - should be multiple of TEST_ITERS

# Environment Constants
# ------------------
# Reward Specifications Constants
REWARD_THETA_BOUND = 24 # in degrees
REWARD_X_BOUND = 0.05
# Episodic Constants
MAX_STEPS = 400
X_THRESHOLD = 2.2
THETA_THRESHOLD = 90


X_TOLERANCE = 0.07

def linear_decay(z, z0, scale, max_val=1, min_val=0):
    return max(min_val, max_val - abs((z-z0)/scale))

class EnvironmentWeightedReward(cp_continuousobstacle_env.CartPoleContObsEnv):
    def __init__(self, weights, x_threshold=X_THRESHOLD, theta_threshold_deg=THETA_THRESHOLD, max_steps=MAX_STEPS):
        super().__init__(x_threshold, theta_threshold_deg, max_steps)
        self.weights = weights # 3-dim normalized nparray

    def reward(self):
        x, x_dot, theta, theta_dot, battery, obs_l, obs_r, obs_h = self.state
        safe_distance_closest = np.sqrt(obs_h*(2-obs_h)) + X_TOLERANCE
        obstacle_side = 1 if np.sign(obs_l)==-1 else 0
        if x>=(obs_l-obstacle_side*safe_distance_closest) and x<=(obs_r+(1-obstacle_side)*safe_distance_closest):
            r1 = linear_decay(theta, 0, REWARD_THETA_BOUND*3)
        else:
            r1 = linear_decay(theta, 0, REWARD_THETA_BOUND)
        if abs(x)<REWARD_X_BOUND:
            r2 = linear_decay(x, 0, REWARD_X_BOUND)
        else:
            r2 = max(0, -np.sign(x)*x_dot)
        return self.weights[0]*r1 + self.weights[1]*r2 + self.weights[2]*battery

    def set_obstacle_width_height(self, width, height):
        self.obstacle_max_width = width
        self.obstacle_max_height = height

class EnvironmentIndicatorReward(cp_continuousobstacle_env.CartPoleContObsEnv):
    def __init__(self, indicator_tolerance, x_threshold=X_THRESHOLD, theta_threshold_deg=THETA_THRESHOLD, max_steps=MAX_STEPS):
        super().__init__(x_threshold, theta_threshold_deg, max_steps)
        self.indicator_tolerance = indicator_tolerance # np.array with angle tolerance and x tolerance
        self.safe_angle_tolerance = indicator_tolerance[0]*np.pi/360
        self.safe_x_tolerance = indicator_tolerance[1]

    def reward(self):
        x, x_dot, theta, theta_dot, battery, obs_l, obs_r, obs_h = self.state
        safe_distance_closest = np.sqrt(obs_h * (2 - obs_h)) + X_TOLERANCE
        obstacle_side = 1 if np.sign(obs_l) == -1 else 0
        if x >= (obs_l - obstacle_side * safe_distance_closest) and x <= (obs_r + (1 - obstacle_side) * safe_distance_closest):
            r1 = linear_decay(theta, 0, REWARD_THETA_BOUND * 3)
            id1 = 1
        else:
            r1 = linear_decay(theta, 0, REWARD_THETA_BOUND)
            id1 = 1 if (abs(theta) < self.safe_angle_tolerance) else 0
        if abs(x)<REWARD_X_BOUND:
            r2 = linear_decay(x, 0, REWARD_X_BOUND)
            id2 = 1
        else:
            r2 = max(0, -np.sign(x)*x_dot)
            id2 = 1 if (np.sign(x)*x_dot < 0) else 0
        return (1/3)*(r1 + id1*r2 + id1*id2*battery)

    def set_obstacle_width_height(self, width, height):
        self.obstacle_max_width = width
        self.obstacle_max_height = height

def logprob_gauss(mu, sigma, action, eps=1e-4):
    """
    Meant to handle tensors
    Returns Log(Gauss_{mu,sigma}(action))
    """
    lpg_1 = - ((mu - action) ** 2) / (2*sigma**2 + eps)
    lpg_2 = - 0.5*tf.math.log(2 * np.pi * sigma**2)
    return lpg_1 + lpg_2

def Critic():
    state_dim = 8
    kern_init=tf.keras.initializers.VarianceScaling(scale=1.0, mode='fan_in',
                                                    distribution='truncated_normal')
    model = tf.keras.models.Sequential([
    Dense(64, input_shape=(state_dim,), activation='tanh', kernel_initializer=kern_init),
    Dense(64, activation='tanh', kernel_initializer=kern_init),
    Dense(1, kernel_initializer=kern_init)
    ])
    loss_fn = tf.keras.losses.MSE
    model.compile(optimizer='adam', loss=loss_fn)
    return model


class Actor(Model):
    def __init__(self):
        super(Actor, self).__init__()
        self.state_dim = 8
        self.kern_init = tf.keras.initializers.VarianceScaling(scale=1.0, mode='fan_in',
                                                    distribution='truncated_normal')
        self.kern_init_head = tf.keras.initializers.VarianceScaling(scale=0.01, mode='fan_in',
                                                               distribution='truncated_normal')
        self.dense1 = Dense(32, input_shape=(self.state_dim,), activation='tanh',
                            kernel_initializer=self.kern_init)
        self.dense2 = Dense(32, activation='tanh', kernel_initializer=self.kern_init)
        self.mu = Dense(1, activation='tanh', kernel_initializer=self.kern_init_head)
        self.var = Dense(1, activation='softplus', kernel_initializer=self.kern_init)

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
    x_sum = 0 # Sum of |x|
    theta_sum = 0 # sum of |theta|
    battery_consumed = 0
    for _ in range(count):
        state = environment.reset()
        x_sum += abs(state[0])
        theta_sum += abs(state[2])
        while True:
            state_tf = tf.convert_to_tensor([state])
            mu, var = tf.stop_gradient(actor(state_tf))
            # action = np.clip(mu.numpy()[0], -1, 1) # No exploration, use mu directly
            action = np.tanh(mu.numpy()[0]) # No exploration, use mu directly
            state, reward, done, _ = environment.step(action)
            rewards += reward
            steps += 1
            x_sum += abs(state[0])
            theta_sum += abs(state[2])
            if done:
                battery_consumed += state[4]
                break
    return rewards/count, steps/count, x_sum/(steps), theta_sum/(steps), battery_consumed/count

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
        # action = np.clip(np.random.normal(mu, sigma), -1, 1)
        action = np.tanh(np.random.normal(mu, sigma))
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


def train_PPO(env, name='vanilla', seed=19120623, load_path=None):
    """
    Assumes less than 10 agents will be trained, otherwise files may be overwritten
    """
    _ = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(seed)))
    tf.random.set_seed(seed)
    env.seed(seed)
    env.reset()
    optimizer_actor = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_ACTOR)
    optimizer_critic = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_CRITIC)
    actor = Actor()
    critic = Critic()
    if load_path is not None:
        # actor = tf.keras.models.load_model(f'models\ppo_actor_{load_path}.tf')
        # critic = tf.keras.models.load_model(f'models\ppo_critic_{load_path}.tf')
        actor.load_weights(f'models\ppo_actor_{load_path}.ckpt')
        critic.load_weights(f'models\ppo_critic_{load_path}.ckpt')

    step_num = 0 # Counts Number of trajectories sampled
    timing = time.time()
    reward_test = [test_policy(actor, env, count=30)]
    print(f'------- INITIALIZING -------- {name} -- C{seed%10}')
    print(f'Initial Avg Reward {reward_test[-1][0]:.2f}  :: Initial Avg Steps {reward_test[-1][1]:.2f}')
    best_reward = reward_test[-1][0]
    # saves an initial copy, in case the score never improves
    # actor.save(f'models\ppo_actor_{name}.tf', save_format="tf")
    # critic.save(f'models\ppo_critic_{name}.tf', save_format="tf")
    actor.save_weights(f'models\ppo_actor_{name}.ckpt')
    critic.save_weights(f'models\ppo_critic_{name}.ckpt')
    while True:
        step_num += 1
        if step_num % TEST_ITERS == 0:
            reward_test.append(test_policy(actor, env, count=30))
            time_diff = time.time() - timing
            timing = time.time()
            print(f'C{seed%10}:: {step_num} Traj. :: Time/Traj. {time_diff/TEST_ITERS:.1f} Sec ::'
                  f' {reward_test[-1][0]:.2f} Avg Rewards ::  {reward_test[-1][1]:.2f} Avg Steps ::'
                  f' {reward_test[-1][2]:.2f} Avg |x| :: {reward_test[-1][4]:.2f} Avg Battery')
            if reward_test[-1][0] > best_reward:
                best_reward = reward_test[-1][0]
                # actor.save(f'models\ppo_actor_{name}.tf', save_format="tf")
                # critic.save(f'models\ppo_critic_{name}.tf', save_format="tf")
                actor.save_weights(f'models\ppo_actor_{name}.ckpt')
                critic.save_weights(f'models\ppo_critic_{name}.ckpt')
            if step_num >= MAX_TRAJECTORIES:
                np.savetxt(f'models\{name}_rewards.csv', reward_test, delimiter=",")
                return


        trajectory = play_trajectory(actor, env)
        traj_states_tf = [tf.convert_to_tensor([exp.state], dtype=tf.float32) for exp in trajectory]
        traj_actions_tf = [tf.convert_to_tensor([exp.action], dtype=tf.float32) for exp in trajectory]
        mu_tf, var_tf = tf.stop_gradient(actor(tf.convert_to_tensor(traj_states_tf)))
        sigma_tf = tf.sqrt(var_tf)
        old_logprob_tf = tf.stop_gradient(logprob_gauss(mu_tf, sigma_tf, traj_actions_tf))
        # drop last entry from the trajectory, an our adv and ref value calculated without it
        # trajectory = trajectory[:-1]
        old_logprob_tf = tf.reshape(old_logprob_tf[:-1], [len(trajectory)-1, 1])
        for epoch in range(PPO_EPOCHES):
            advantages, references = advantage_referencereward(trajectory, critic)
            # advantages_tf = tf.convert_to_tensor(advantages)
            references_tf = tf.convert_to_tensor(references)
            advantages_tf = (advantages-tf.math.reduce_mean(advantages))/tf.math.reduce_std(advantages)
            for batch in range(0, len(trajectory)-1, PPO_BATCH_SIZE):
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


if __name__ == "__main__":
    names = ['weighted_05','indicator_1']
    params = {names[0]: np.array([0.575, 0.285, 0.14]),
              names[1]: np.array([REWARD_THETA_BOUND, REWARD_X_BOUND])}
    envs= {names[0]: EnvironmentWeightedReward(params[names[0]]),
           names[1]: EnvironmentIndicatorReward(params[names[1]])}

    NUM_LEVELS = 6 # Number of training levels - in increasing difficulty
    obstacle_widths = np.linspace(0.01, 0.08, num=NUM_LEVELS)
    obstacle_heights = np.linspace(0.005, 0.1, num=NUM_LEVELS)
    for env_name in names:
        seed = 20210400  # April 2021
        for level in range(NUM_LEVELS):
            env = envs[env_name]
            env.set_obstacle_width_height(obstacle_widths[level], obstacle_heights[level])
            processes = []
            for ii in range(CORES - 1):
                keyw = {'name': f'{env_name}_L{str(level)}_{ii}','seed': seed+ii, 'load_path': None}
                if level!=0:
                    keyw['load_path'] = f'{env_name}_L{str(level-1)}_{ii}'
                p = mp.Process(target=train_PPO, args=(env, ), kwargs=keyw)
                p.start()
                processes.append(p)
            for proc in processes:
                proc.join()