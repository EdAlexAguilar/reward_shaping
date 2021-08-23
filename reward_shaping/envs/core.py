from abc import ABC, abstractmethod
from typing import Dict, Any, List

import gym


class RewardFunction(ABC):
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def __call__(self, state, action=None, next_state=None, info=None) -> float:
        pass


class RewardWrapper(gym.Wrapper):

    def __init__(self, env: gym.Env, reward_fn: RewardFunction):
        super().__init__(env)
        self._state = None
        self._reward = 0.0
        self._return = 0.0
        self._reward_fn = reward_fn

    def reset(self, **kwargs):
        state = self.env.reset(**kwargs)
        self._state = state
        self._reward = 0.0
        self._return = 0.0
        return state

    def step(self, action: Any):
        next_state, _, done, info = self.env.step(action)
        reward = self._reward_fn(state=self._state, action=action, next_state=next_state, info=info)
        self._state = next_state
        self._reward = reward
        self._return += reward
        return next_state, reward, done, info

    def render(self, mode='human', **kwargs):
        return super(RewardWrapper, self).render(mode=mode, info={'reward': self._reward, 'return': self._return})


class STLRewardConfig(ABC):
    def __init__(self, **kwargs):
        pass

    @property
    @abstractmethod
    def monitoring_variables(self) -> List[str]:
        """List of monitored variables (ie, the variables which occur in the stl spec)."""
        pass

    @property
    @abstractmethod
    def monitoring_types(self) -> List[str]:
        """List of variables types."""
        pass

    @property
    @abstractmethod
    def spec(self) -> str:
        """
        stl specification used for evaluation
        """
        pass

    @abstractmethod
    def get_monitored_state(self, state, done, info):
        """Given observed quantities (eg, state,info..), prepare the variable monitored in the stl spec."""
        pass


class STLRewardWrapper(gym.Wrapper):
    """ This is an 'episodic' wrapper which evaluate a spec in the terminal states."""

    def __init__(self, env: gym.Env, stl_conf: STLRewardConfig):
        super(STLRewardWrapper, self).__init__(env)
        self._episode = {var: [] for var in stl_conf.monitoring_variables}
        self._stl_conf = stl_conf
        self._reward = 0.0
        self._return = 0.0

    def reset(self, **kwargs):
        self._episode = {var: [] for var in self._stl_conf.monitoring_variables}
        self._reward = 0.0
        self._return = 0.0
        state = self.env.reset(**kwargs)
        return state

    def _compute_episode_robustness(self):
        import rtamt
        spec = rtamt.STLSpecification()
        for v, t in zip(self._stl_conf.monitoring_variables, self._stl_conf.monitoring_types):
            spec.declare_var(v, f'{t}')
        spec.spec = self._stl_conf.spec
        try:
            spec.parse()
        except rtamt.STLParseException:
            return
        # preprocess format, evaluate, post process
        robustness_trace = spec.evaluate(self._episode)
        return robustness_trace[0][1]

    def step(self, action):
        obs, _, done, info = super().step(action)
        # prepare monitoring variables, collect them
        monitored_state = self._stl_conf.get_monitored_state(obs, done, info)
        for key, value in monitored_state.items():
            self._episode[key].append(value)
        # evaluate reward only in terminal states
        reward = self._compute_episode_robustness() if done else 0.0
        self._reward = reward
        self._return += reward
        return obs, reward, done, info

    def render(self, mode='human', **kwargs):
        return super(STLRewardWrapper, self).render(mode=mode, info={'reward': self._reward, 'return': self._return})
