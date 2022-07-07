from reward_shaping.envs.racecar.multi_agent_racecar_env import MultiAgentRacecarEnv

# parameters
n_episodes = 100     # n of episodes to simulate
render = True       # enable rendering
eval = False         # if true, start cars on grid position; if false, random (reasonably close) position

# define scenario files
scenario_files = [                      # if many files, many environment are created to increase diversity
    "treitlstrasse_multi_agent.yml",    # for fast prototyping, you can comment out any of them
    "columbia_multi_agent.yml"
]
env = MultiAgentRacecarEnv(scenario_files, render=render, eval=eval)

print()
print("[observation space]")
print(env.observation_space)
print()

print()
print("[action space]")
print(env.action_space)
print()

# run simulations
for i in range(n_episodes):
    env.reset()
    done = False
    while not done:
        # car1: go straight full speed (note: actions are normalized in +-1)
        action_car1 = {
            "speed": 1.0, "steering": 0.0
        }
        # car2: drive randomly
        action_car2 = env.action_space['B'].sample()

        # put actions in dictionary
        actions = {
            "A": action_car1,
            "B": action_car2,
        }

        # simulate
        obss, rewards, dones, infos = env.step(actions)
        done = any(dones.values())  # since many agents, we need to aggregate the termination condition

env.close()
print("[info] done")