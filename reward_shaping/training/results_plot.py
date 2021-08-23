import numpy as np
import matplotlib.pyplot as plt
import ppo_mp as ppo

# names = ['weighted_066','weighted_05','indicator_1','indicator_066']
# formats = {names[0]: '-k', names[1]: '-r', names[2]: '--b', names[3]: '--g'}

# names = ['weighted_05','indicator_1']
# formats = {names[0]: '-k', names[1]: '--b'}
names = ['indicator_1']
formats = {names[0]: '--b'}
num_agents = int(ppo.CORES)

level="L2"

rewards = {}
for name in names:
    rewards[name] = 0
    for ii in range(num_agents):
        name_ii = f'{name}_{level}_{str(ii)}'
        rewards[name_ii] = np.genfromtxt(f'models\{name_ii}_rewards.csv', delimiter=',')
        rewards[name] += rewards[name_ii]
    rewards[name] = rewards[name]/8

# for _, name in enumerate(names):
#   rewards[name] = np.genfromtxt(f'{name}_rewards.csv', delimiter=',')

env_calls = np.linspace(0.0, ppo.MAX_TRAJECTORIES*ppo.TRAJECTORY_SIZE, int(ppo.MAX_TRAJECTORIES/ppo.TEST_ITERS + 1)) # x-axis

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(5, 3))

for name in names:
    axes[0,0].plot(env_calls, rewards[name][:, 0], formats[name], label=name) # rewards
    axes[0,1].plot(env_calls,30* rewards[name][:, 2], formats[name], label=name) # |x|
    axes[1,0].plot(env_calls, 30*360*rewards[name][:, 3]/np.pi, formats[name], label=name) # |theta|
    axes[1,1].plot(env_calls, rewards[name][:, 4], formats[name], label=name) # battery

# fig.tight_layout()

# LABELS AND TITLES
axes[0,0].set_xlabel('Environment Steps')
axes[0,1].set_xlabel('Environment Steps')
axes[1,0].set_xlabel('Environment Steps')
axes[1,1].set_xlabel('Environment Steps')

axes[0,0].set_title('(Self) Reward')
axes[0,1].set_title('Avg |x| deviation')
axes[1,0].set_title('Avg |theta| deviation')
axes[1,1].set_title('Avg Left-over battery')

# plt.legend(f'Weighted Reward w1*r1 + w2*r2  ; w=(0.6,0.4)')
plt.legend()
plt.show()
