import collections
from typing import Dict, Tuple, Any

import gym
from gym.spaces import Box
from gym.wrappers import LazyFrames
import numpy as np
from stable_baselines3.common.type_aliases import GymObs


class FixResetWrapper(gym.Wrapper):
    """Fix a reset mode to sample initial condition from starting grid or randomly over the track."""

    def __init__(self, env, mode):
        assert mode in ['grid', 'random', 'random_bidirectional']
        self._mode = mode
        super(FixResetWrapper, self).__init__(env)

    def reset(self):
        return super(FixResetWrapper, self).reset(mode=self._mode)


class FilterObservationWrapper(gym.Wrapper):
    """
    observation wrapper that filter a single observation and return is without dictionary,
    all the observable quantities are moved to the info as `state`
    """

    def __init__(self, env, obs_list=[]):
        super(FilterObservationWrapper, self).__init__(env)
        self._obs_list = obs_list
        self.observation_space = gym.spaces.Dict({obs: self.env.observation_space[obs] for obs in obs_list})

    def _filter_obs(self, original_obs):
        new_obs = {}
        for obs in self._obs_list:
            assert obs in original_obs
            new_obs[obs] = original_obs[obs]
        return new_obs

    def step(self, action):
        original_obs, reward, done, info = super().step(action)
        new_obs = self._filter_obs(original_obs)
        # add original state into the info
        new_info = info
        new_info['state'] = {name: value for name, value in original_obs.items()}
        return new_obs, reward, done, info

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        new_obs = self._filter_obs(obs)
        return new_obs


class FlattenAction(gym.ActionWrapper):
    """Action wrapper that flattens the action."""

    def __init__(self, env):
        super(FlattenAction, self).__init__(env)
        self.action_space = gym.spaces.utils.flatten_space(self.env.action_space)

    def action(self, action):
        return gym.spaces.utils.unflatten(self.env.action_space, action)

    def reverse_action(self, action):
        return gym.spaces.utils.flatten(self.env.action_space, action)


class FrameStackOnChannel(gym.Wrapper):
    r"""
    Observation wrapper that stacks the observations in a rolling manner.

    Implementation from gym.wrappers but squeeze observation (then removing channel dimension),
    and support dictionary
    """

    def __init__(self, env, num_stack):
        super(FrameStackOnChannel, self).__init__(env)
        self.num_stack = num_stack

        self.frames = {
            k: collections.deque(maxlen=num_stack) for k in self.observation_space.spaces
        }
        lows = {
            k: np.repeat(space.low, num_stack, axis=0) for k, space in
            self.observation_space.spaces.items()
        }
        highs = {
            k: np.repeat(space.high, num_stack, axis=0) for k, space in
            self.observation_space.spaces.items()
        }

        self.observation_space = gym.spaces.Dict(
            {
                k: gym.spaces.Box(lows[k], highs[k], dtype=self.observation_space.spaces[k].dtype)
                for k in self.observation_space.spaces
            }
        )

    def _get_observation(self):
        assert all([len(frames) == self.num_stack for k, frames in self.frames.items()]), self.frames
        # return {k: LazyFrames(list(self.frames[k]), self.lz4_compress) for k in self.frames}
        return {k: np.array(self.frames[k], dtype=np.float32) for k in self.frames}

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        for k in self.observation_space.spaces:
            self.frames[k].append(np.squeeze(observation[k]))  # assume 1d channel dimension and remove it
        return self._get_observation(), reward, done, info

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        for k in observation:
            [self.frames[k].append(np.squeeze(observation[k])) for _ in range(self.num_stack)]
        return self._get_observation()


class FixSpeedControl(gym.ActionWrapper):
    """
    reduce the original action space, and fix the speed command to a constant value
    """

    def __init__(self, env, fixed_speed: float = 2.0):
        super(FixSpeedControl, self).__init__(env)
        assert type(fixed_speed) == float
        assert "speed" in self.action_space.spaces.keys(), "the action space does not have any 'speed' action"
        self._fixed_speed = np.array([fixed_speed])
        assert self.action_space["speed"].contains(
            self._fixed_speed), "the fixed speed is not in the original action space"
        self.action_space = gym.spaces.Dict(
            {a: space for a, space in self.action_space.spaces.items() if a != "speed"})

    def action(self, action):
        action["speed"] = self._fixed_speed
        return action


class NormalizeObservationWithMinMax(gym.ObservationWrapper):
    """
    Normalize Box observations with respect to given min and max values.
    The observations to be normalized are given as a dictionary: obs_name -> (minvalue, maxvalue)

    The values are clipped w.r.t. min/max values to avoid unexpected values (e.g., <-1 or >1)
    """

    def __init__(self, env, map_name2values: Dict[str, Tuple[float, float]]):
        super(NormalizeObservationWithMinMax, self).__init__(env)
        self._map_name2values = map_name2values
        for obs_name in self._map_name2values.keys():
            self.observation_space[obs_name] = gym.spaces.Box(low=-1, high=+1,
                                                              shape=self.observation_space[obs_name].shape)

    def observation(self, observation):
        for obs_name, (min_value, max_value) in self._map_name2values.items():
            new_obs = np.clip(observation[obs_name], min_value, max_value)  # clip it
            new_obs = (new_obs - min_value) / (max_value - min_value)  # norm in 0,1
            observation[obs_name] = -1 + 2 * new_obs  # finally, map it to -1..+1
        return observation


class FrameSkip(gym.Wrapper):
    """
    Adapted from https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/atari_wrappers.py
    """

    def __init__(self, env: gym.Env, skip: int = 4):
        gym.Wrapper.__init__(self, env)
        self._skip = skip

    def step(self, action: int) -> GymObs:
        """
        Step the environment with the given action
        Repeat action, sum reward, and max over last observations.
        :param action: the action
        :return: observation, reward, done, information
        """
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info

    def reset(self, **kwargs) -> GymObs:
        return self.env.reset(**kwargs)


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
        self.npc_obs = npc_obs  # save current obs for npc next step
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
        delta_speed = action["delta_speed"][0] * self._max_delta_speed  # ranges in +-max delta speed
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
        assert all([a in self.action_space.keys() for a in
                    ["steering", "speed"]]), "invalid action space in action-history wrapper"

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
        self._last_obss = collections.deque([np.zeros(self._original_shape, dtype=np.float32)] * self._n_last_obs,
                                            maxlen=self._n_last_obs)
        obs = super(ObservationHistoryWrapper, self).reset(**kwargs)
        return obs

    def observation(self, observation):
        self._last_obss.append(observation[self._obs_name])
        observation[self._new_obs_name] = np.array(self._last_obss, dtype=np.float32)
        return observation
