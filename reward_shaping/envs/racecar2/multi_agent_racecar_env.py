import collections
import os
from typing import List, Dict, Any
import numpy as np

import racecar_gym
from racecar_gym import MultiAgentScenario
from racecar_gym.agents import FollowTheGap, FollowTheWall
from racecar_gym.agents.random import RandomAgent
from racecar_gym.envs.gym_api import ChangingTrackMultiAgentRaceEnv

NPCs = {
    "ftg": FollowTheGap,
    "ftw": FollowTheWall,
    "rnd": RandomAgent,
}


class MultiAgentRacecarEnv(ChangingTrackMultiAgentRaceEnv):

    def __init__(self,
                 scenario_files: List[str],
                 order: str = 'sequential',
                 npc: str = 'ftg',
                 npc_params: Dict[str, Any] = {},
                 **kwargs):
        # make race environment
        params = self._get_params(**kwargs)
        scenarios = [MultiAgentScenario.from_spec(path=str(f"{os.path.dirname(__file__)}/config/{sf}"),
                                                  rendering=params["render"]) for sf in scenario_files]
        super(MultiAgentRacecarEnv, self).__init__(scenarios=scenarios, order=order)

        # spec params
        self._max_steps = params["max_steps"]
        self._steps = 0
        self._initial_progress = None

        self._eval = params["eval"]
        self._seed = params["seed"]
        self.seed(self._seed)

        # internal variables
        assert len(self.scenario.agents) == 2, "not supported more than 2 agents in the scenario"
        agents_ids = list(self.scenario.agents.keys())
        self._agent_id, self._npc_id = agents_ids[0], agents_ids[1]
        self._npc = NPCs[npc](npc_params)
        self._npc_obs = None
        self._npc_state = None

        assert all([self.observation_space[self._agent_id] == self.observation_space[i] for i in
                    self.scenario.agents]), "all agents must have the same obs space"
        assert all([self.action_space[self._agent_id] == self.action_space[i] for i in
                    self.scenario.agents]), "all agents must have the same action space"
        self.observation_space = self.observation_space[self._agent_id]
        self.action_space = self.action_space[self._agent_id]

    @staticmethod
    def _get_params(**kwargs):
        params = {
            "max_steps": 600,
            "render": False,
            "eval": False,
            "seed": 0,
        }
        for k, v in kwargs.items():
            params[k] = v
        return params

    def reset(self):
        joint_obs = super(MultiAgentRacecarEnv, self).reset(mode='grid' if self._eval else 'random_ball')
        # save obs npc for later step
        self._npc_obs = joint_obs[self._npc_id]
        self._npc_state = self._npc.reset()
        # reset roles
        self._agent_id = "A"  # todo: eventually sample
        self._npc_id = "B"  # todo: eventually sample
        agent_obs = joint_obs[self._agent_id]
        # interval vars
        self._initial_progress = None
        self._steps = 0
        return agent_obs

    def step(self, action: Dict):
        # perform sim step
        npc_action, self._npc_state = self._npc.get_action(self._npc_obs, self._npc_state)
        joint_action = {self._agent_id: action, self._npc_id: npc_action}
        all_joint_returns = super(MultiAgentRacecarEnv, self).step(joint_action)
        # unpack all joint quantities and keep only agent ones (e.g., joint_obs -> agent_obs)
        obs, reward, done, info = [joint_return[self._agent_id] for joint_return in all_joint_returns]
        # update internal variables
        self._steps += 1
        self._npc_obs = all_joint_returns[0][self._npc_id]  # keep last npc observation for next step
        return obs, reward, done, info

    def _extend_obs(self, obs, info):
        # todo
        return obs

    def _extend_info(self, obs, info):
        # todo
        return info

    def _check_termination(self, obs, done, info):
        # todo
        return bool(done)

    def render(self, mode):
        view_mode = "follow"
        screen = super(MultiAgentRacecarEnv, self).render(mode=view_mode)
        if mode == "rgb_array":
            return screen


def test_npc_controllers():
    scenario_files = ["columbia_multi_agent.yml"]

    npc_params = {
        "ftg": {"scan_field": "lidar_64"},
        "ftw": {"scan_field": "lidar_64", "scan_size": 64},
        "rnd": {},
    }

    n_episodes = 3

    for npc in npc_params.keys():
        agent = NPCs[npc](npc_params[npc])
        env = MultiAgentRacecarEnv(scenario_files, npc=npc, npc_params=npc_params[npc], render=False)

        progresses = []

        for ep in range(n_episodes):
            obs = env.reset()
            done = False

            action, state = agent.get_action(obs)
            obs, reward, done, info = env.step(action)
            init_progress = info["progress"]

            while not done:
                action, state = agent.get_action(obs)
                obs, reward, done, info = env.step(action)
                if done:
                    progress = info['lap'] - 1 + info['progress'] - init_progress
                    print(f"\t[info] npc: {npc}, episode {ep}: progress: {progress}")
                    progresses.append(progress)
        env.close()

        print(f"[info] npc: {npc}, avg progress {np.mean(progresses)}\n")
    assert True


if __name__ == "__main__":
    test_npc_controllers()
