import collections
import os
from typing import List, Dict, Any
import numpy as np

import racecar_gym
from gym.spaces import Box
from racecar_gym import MultiAgentScenario
from racecar_gym.agents import FollowTheGap, FollowTheWall
from racecar_gym.agents.follow_the_wall import PID
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
        self._track_length = params["track_length"]
        self._safety_distance = params["reward_params"]["safety_distance"]
        self._min_comfort_distance = params["reward_params"]["min_comfort_distance"]
        self._max_comfort_distance = params["reward_params"]["max_comfort_distance"]
        self._target_progress = params["reward_params"]["target_progress"]
        self._target_dist2obst = params["reward_params"]["target_dist2obst"]
        self._comfort_max_steering = params["reward_params"]["comfort_max_steering"]
        self._comfort_max_norm = params["reward_params"]["comfort_max_norm"]
        self._min_velx = params["reward_params"]["min_velx"]
        self._max_velx = params["reward_params"]["max_velx"]
        self._limit_velx = params["action_config"]["max_velx"]
        self._max_steps = params["max_steps"]
        self._frame_skip = params["frame_skip"]
        self._steps = 0
        self._initial_progress = None

        self._eval = params["eval"]
        self._seed = params["seed"]
        self.seed(self._seed)

        # internal variables
        assert len(self.scenario.agents) == 2, "not supported more than 2 agents in the scenario"
        agents_ids = list(self.scenario.agents.keys())
        self._agent_id, self._npc_id = agents_ids[0], agents_ids[1]
        self._npc_type = npc
        self._npc_params = npc_params
        self._npc = NPCs[self._npc_type](self._npc_params)
        self._npc_obs = None
        self._npc_state = None
        self._npc_min_base_speed = params["npc_min_base_speed"]
        self._npc_max_base_speed = params["npc_max_base_speed"]
        self._npc_min_var_speed = params["npc_min_var_speed"]
        self._npc_max_var_speed = params["npc_max_var_speed"]
        if self._npc_type == "ftw":
            self._npc_min_dist_left = params["npc_min_dist_left"]
            self._npc_max_dist_left = params["npc_max_dist_left"]

        # reduce observation space to 1 agent
        # extend it with state information (we need them to compute the potential, the agent do not directly observe them)
        assert all([self.observation_space[self._agent_id] == self.observation_space[i] for i in
                    self.scenario.agents]), "all agents must have the same obs space"
        assert all([self.action_space[self._agent_id] == self.action_space[i] for i in
                    self.scenario.agents]), "all agents must have the same action space"
        self.observation_space = self.observation_space[self._agent_id]
        self.action_space = self.action_space[self._agent_id]

        self.observation_space["progress"] = Box(low=0.0, high=1.0, dtype=np.float32, shape=(1,))
        self.observation_space["dist2obst"] = Box(low=0.0, high=1.0, dtype=np.float32, shape=(1,))
        self.observation_space["collision"] = Box(low=0.0, high=1.0, dtype=np.float32, shape=(1,))
        minvel, maxvel = self.observation_space["velocity"].low[0], self.observation_space["velocity"].high[0]
        self.observation_space["velocity_x"] = Box(low=minvel, high=maxvel, shape=(1,))
        self.observation_space["dist_ego2npc"] = Box(low=-np.Inf, high=np.Inf, shape=(1,))
        self.observation_space["remaining_time"] = Box(low=0.0, high=1.0, shape=(1,))

    @staticmethod
    def _get_params(**kwargs):
        params = {
            "max_steps": 2500,
            "track_length": 38.45,  # compute in oval track
            "reward_params": {
                "safety_distance": -0.75,
                "min_comfort_distance": -2.00,
                "max_comfort_distance": -1.50,
                "target_progress": 0.98,
                "target_dist2obst": 0.5,
                "min_velx": 2.0,
                "max_velx": 3.0,
                "comfort_max_steering": 0.1,
                "comfort_max_norm": 0.25,
            },
            "action_config": {
                "delta_speed": False,
                "min_velx": 0.0,
                "max_velx": +3.5,
                "cap_min_velx": 0.0,
                "cap_max_velx": +3.5,
                "max_accx": 4.0,
                "dt": 0.01,
            },
            "npc": "ftg",
            "npc_min_base_speed": 1.25,
            "npc_max_base_speed": 1.75,
            "npc_min_var_speed": 0.25,
            "npc_max_var_speed": 0.75,
            "npc_params": {
                "scan_field": "lidar_64",
                # longitudinal control
                "base_speed": 1.75,
                "variable_speed": 0.75,
            },
            "frame_skip": 1,
            "render": False,
            "eval": False,
            "seed": 0,
        }
        for k, v in kwargs.items():
            params[k] = v
        return params

    def reset(self):
        joint_obs = super(MultiAgentRacecarEnv, self).reset(mode='grid' if self._eval else 'random_ball')
        # perform dummy step to obtain agents' infos
        no_action = {"speed": -1.0, "steering": 0.0}
        joint_no_action = {"A": no_action, "B": no_action}
        joint_obs, joint_reward, joint_done, joint_info = super(MultiAgentRacecarEnv, self).step(joint_no_action)
        # assign roles
        is_a_in_front = joint_info["A"]["progress"] > joint_info["B"]["progress"]
        self._npc_id = "A" if is_a_in_front else "B"
        self._agent_id = "B" if is_a_in_front else "A"
        agent_obs = self._extend_obs(joint_obs, joint_info)
        # save obs npc for later step
        self._npc_obs = joint_obs[self._npc_id]
        new_npc_params = self._randomize_npc_params(self._npc_type, self._npc_params)
        self._npc_state = self._npc.reset(config=new_npc_params)
        # interval vars
        self._initial_progress = None
        self._steps = 0
        return agent_obs

    def _randomize_npc_params(self, npc_type, default_params):
        # ftg: randomize velocity profile
        # ftw: randomize velocity profile and distance to wall
        default_params["base_speed"] = self._npc_min_base_speed + np.random.rand() * (
                    self._npc_max_base_speed - self._npc_min_base_speed)
        default_params["variable_speed"] = self._npc_min_var_speed + np.random.rand() * (
                    self._npc_max_var_speed - self._npc_min_var_speed)
        if npc_type == "ftw":
            default_params["target_distance_left"] = self._npc_min_dist_left + np.random.rand() * (
                    self._npc_max_dist_left - self._npc_min_dist_left)
        return default_params

    def step(self, action: Dict):
        # perform sim step
        npc_action, self._npc_state = self._npc.get_action(self._npc_obs, self._npc_state)
        joint_action = {self._agent_id: action, self._npc_id: npc_action}
        joint_obs, joint_reward, joint_done, joint_info = super(MultiAgentRacecarEnv, self).step(joint_action)
        # unpack all joint quantities and keep only agent ones (e.g., joint_obs -> agent_obs)
        obs = self._extend_obs(joint_obs, joint_info)
        reward = joint_reward[self._agent_id]
        done = self._check_termination(obs, joint_done, joint_info)
        info = self._extend_info(reward, done, joint_info)
        # update internal variables
        self._steps += 1
        self._npc_obs = joint_obs[self._npc_id]  # keep last npc observation for next step
        return obs, reward, done, info

    def _extend_obs(self, joint_obs, joint_info):
        obs = joint_obs[self._agent_id]
        info, info_npc = joint_info[self._agent_id], joint_info[self._npc_id]
        if self._initial_progress is None and info["progress"] is not None:
            # update the initial-progress on the first available progress after reset
            self._initial_progress = info["progress"]
        progress = 0.0 if self._initial_progress is None else (info["lap"] - 1) + (
                    info["progress"] - self._initial_progress)
        obs["collision"] = float(info["wall_collision"])
        obs["progress"] = progress
        obs["remaining_time"] = 1.0 if self._eval else (self._max_steps - self._steps)/self._max_steps
        obs["dist2obst"] = info["obstacle"]
        obs["velocity_x"] = np.array([obs["velocity"][0]], dtype=np.float32)
        obs["dist_ego2npc"] = ((info["lap"] + info["progress"]) - (info_npc["lap"] + info_npc["progress"])) * self._track_length
        return obs

    def _extend_info(self, reward, done, joint_info):
        info = joint_info[self._agent_id]
        info["default_reward"] = reward
        info["safety_distance"] = self._safety_distance
        info["min_comfort_distance"] = self._min_comfort_distance
        info["max_comfort_distance"] = self._max_comfort_distance
        info["target_progress"] = self._target_progress
        info["target_dist2obst"] = self._target_dist2obst
        info["comfort_max_steering"] = self._comfort_max_steering
        info["comfort_max_norm"] = self._comfort_max_norm
        info["min_velx"] = self._min_velx
        info["max_velx"] = self._max_velx
        info["limit_velx"] = self._limit_velx
        info["steps"] = self._steps
        info["max_steps"] = self._max_steps
        info["frame_skip"] = self._frame_skip
        info["done"] = done
        return info

    def _check_termination(self, obs, joint_done, joint_info):
        info, info_npc = joint_info[self._agent_id], joint_info[self._npc_id]
        collision = info["wall_collision"] or len(info["opponent_collisions"]) > 0
        break_safety_dist = not (obs["dist_ego2npc"] < self._safety_distance)
        lap_completion = obs["progress"] >= self._target_progress
        timeout = self._steps >= self._max_steps
        return bool(collision or break_safety_dist or lap_completion or timeout)

    def render(self, mode):
        view_mode = "follow"
        screen = super(MultiAgentRacecarEnv, self).render(mode=view_mode, agent=self._agent_id)
        if mode == "rgb_array":
            return screen


def test_npc_controllers():
    scenario_files = ["oval_multi_agent.yml"]

    from racecar_gym.agents.follow_the_wall import PID
    npc_params = {
        "ftg": {"scan_field": "lidar_64"},
        "ftw": {"scan_field": "lidar_64",
                "scan_size": 64,
                # lateral control
                "target_distance_left": 0.5,
                "max_deviation": 0.5,
                "pid_config": PID.PIDConfig(2.0, 0.0, 0.1),
                # longitudinal control
                "base_speed": 1.75,
                "variable_speed": 0.75,
                },
        "rnd": {},
    }

    n_episodes = 30

    for npc in ["ftg"]:  # npc_params.keys():
        agent = NPCs["ftg"](npc_params["ftg"])
        env = MultiAgentRacecarEnv(scenario_files, npc=npc, npc_params=npc_params[npc], render=True)

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

            progress = info['lap'] - 1 + info['progress'] - init_progress
            print(f"\t[info] npc: {npc}, episode {ep}: progress: {progress}")
            progresses.append(progress)
        env.close()

        print(f"[info] npc: {npc}, avg progress {np.mean(progresses)}\n")
    assert True


if __name__ == "__main__":
    scenario_files = ["oval_multi_agent.yml"]

    npc = "ftg"
    npc_params = {"scan_field": "lidar_64"}
    agent = NPCs["ftg"](npc_params)
    env = MultiAgentRacecarEnv(scenario_files,
                               npc=npc, npc_params=npc_params,
                               render=True, eval=True)

    n_episodes = 10

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

        progress = info['lap'] - 1 + info['progress'] - init_progress
        print(f"\t[info] npc: {npc}, episode {ep}: progress: {progress}")
        progresses.append(progress)
    env.close()
