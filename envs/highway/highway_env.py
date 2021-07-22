from typing import Dict, Any

from highway_env.envs import HighwayEnv, Action, Tuple
from highway_env.envs.common.abstract import Observation

from envs.monitorable_env import MonitorableEnv


class CustomHighwayEnv(MonitorableEnv, HighwayEnv):

    @property
    def train_spec(self) -> Tuple[str, str]:
        pass

    @property
    def eval_spec(self) -> Tuple[str, str]:
        pass

    def __init__(self, config, eval=False, seed=0):
        super(CustomHighwayEnv, self).__init__()
        self.config = config
        self.eval = eval  # todo: if collision make deterministic the starting points? other ideas?
        self.seed(seed)
        # monitoring
        self.last_complete_episode = None
        self.last_cont_spec = None
        self.last_bool_spec = None

    @property
    def monitoring_variables(self):
        return ['time', 'dist_npc', 'collision']

    @property
    def monitoring_types(self):
        return ['int', 'float', 'float']

    def get_monitored_state(self, obs, done, info) -> Dict[str, Any]:
        monitored_state = {'dist_npc': None,  # todo: find a way to measure distances to npcs
                           'collision': self.vehicle.crashed}
        return monitored_state

    def step(self, action: Action) -> Tuple[Observation, float, bool, dict]:
        obs, reward, done, info = super(CustomHighwayEnv, self).step(action)
        monitored_state = self.get_monitored_state(obs, done, info)
        self.expand_episode(monitored_state, done)
        return obs, reward, done, info


