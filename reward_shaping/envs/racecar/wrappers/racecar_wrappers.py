from typing import Dict, Any

import gym
import numpy as np
from gym.spaces import Box
import collections

class Multi2SingleEnv(gym.Wrapper):
    """Puts a safety layer between agent actions and environment"""
    def __init__(self, env=None, agent_name='B', npc_name='A', npc_controller=None):
        self.agent = agent_name
        self.npc = npc_name
        self.npc_controller = npc_controller
        assert npc_controller is not None
        super(Multi2SingleEnv, self).__init__(env)
        self.observation_space = env.observation_space[self.agent]
        self.action_space = env.action_space[self.agent]

    def collect_actions(self, agent_a, npc_a):
        return {self.agent: agent_a, self.npc: npc_a}

    def split(self, joint):
        """
        :param joint: Dictionary with keys being self.agent and self.npc
        :return: split dictionary
        """
        return joint[self.agent], joint[self.npc]

    def reset(self):
        joint_obs = self.env.reset()
        agent_obs, npc_obs = self.split(joint_obs)
        self.npc_obs = npc_obs
        return agent_obs

    def step(self, action):
        npc_a = self.npc_controller.act(self.npc_obs)
        joint_action = self.collect_actions(action, npc_a)
        joint_obs, joint_rew, joint_done, joint_info = self.env.step(joint_action)
        agent_obs, npc_obs = self.split(joint_obs)
        agent_rew, _ = self.split(joint_rew)
        done = any(joint_done.values())
        agent_info, npc_info = self.split(joint_info)
        self.npc_obs = npc_obs # save current obs for npc next step
        return agent_obs, agent_rew, done, agent_info


class DeltaSpeedWrapper(gym.ActionWrapper):
    def __init__(self, env, frame_skip: int, action_config: Dict[str, Any], **kwargs):
        super(DeltaSpeedWrapper, self).__init__(env)
        self.action_space = gym.spaces.Dict({
            "delta_speed": Box(low=-1.0, high=1.0, dtype=np.float32, shape=(1,)),
            "steering": Box(low=-1.0, high=1.0, dtype=np.float32, shape=(1,))
        })
        self._max_delta_speed = action_config["max_accx"] * frame_skip * action_config["dt"]
        self._minspeed = action_config["min_velx"]
        self._maxspeed = action_config["max_velx"]
        self._cap_minspeed = action_config["cap_min_speed"]
        self._cap_maxspeed = action_config["cap_max_speed"]

    def reset(self, **kwargs):
        self.speed_ms = np.array([0.0], dtype=np.float32)  # last speed cmd in m/s
        return super(DeltaSpeedWrapper, self).reset(**kwargs)

    def action(self, action):
        delta_speed = action["delta_speed"][0] * self._max_delta_speed     # ranges in +-max delta speed
        self.speed_ms = self.speed_ms + delta_speed
        self.speed_ms = np.clip(self.speed_ms, self._cap_minspeed, self._cap_maxspeed)

        norm_speed = -1 + 2 * (self.speed_ms - self._minspeed) / (self._maxspeed - self._minspeed)

        original_action = {
            "speed": norm_speed,
            "steering": action["steering"],
        }
        return original_action

    def reverse_action(self, action):
        raise NotImplementedError("delta speed wrapper reverse action")
        pass


class ActionHistoryWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_last_actions: int, low: float = -1, high: float = +1, **kwargs):
        super(ActionHistoryWrapper, self).__init__(env)
        self._n_last_actions = n_last_actions

        shape = (self._n_last_actions, len(self.action_space))
        self.observation_space["last_actions"] = Box(low=low, high=high, shape=shape)
        assert all([a in self.action_space.keys() for a in ["steering", "speed"]]), "invalid action space in action-history wrapper"

    def reset(self, **kwargs):
        self._last_actions = collections.deque([[0.0] * len(self.action_space)] * self._n_last_actions,
                                               maxlen=self._n_last_actions)
        return super(ActionHistoryWrapper, self).reset(**kwargs)

    def step(self, action: Dict[str, float]):
        flat_action = np.array([action["steering"][0], action["speed"][0]])
        self._last_actions.append(flat_action)
        return super(ActionHistoryWrapper, self).step(action)

    def observation(self, observation):
        observation["last_actions"] = np.array(self._last_actions, dtype=np.float32)
        return observation


class ObservationHistoryWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_last_observations: int, obs_name: str, **kwargs):
        super(ObservationHistoryWrapper, self).__init__(env)
        assert obs_name in self.observation_space.keys(), "invalid observation in observation-history wrapper"

        self._n_last_obs = n_last_observations
        self._obs_name = obs_name
        self._new_obs_name = f"last_{obs_name}"

        self._original_shape = self.observation_space[obs_name].shape
        self._new_shape = (self._n_last_obs, *self._original_shape)
        low = np.tile(self.observation_space[obs_name].low, reps=(self._n_last_obs, 1))
        high = np.tile(self.observation_space[obs_name].high, reps=(self._n_last_obs, 1))
        self.observation_space[self._new_obs_name] = Box(low=low, high=high, shape=self._new_shape)

    def reset(self, **kwargs):
        self._last_obss = collections.deque([np.zeros(self._original_shape, dtype=np.float)] * self._n_last_obs,
                                            maxlen=self._n_last_obs)
        obs = super(ObservationHistoryWrapper, self).reset(**kwargs)
        return obs

    def observation(self, observation):
        self._last_obss.append(observation[self._obs_name])
        observation[self._new_obs_name] = np.array(self._last_obss, dtype=np.float32)
        return observation


