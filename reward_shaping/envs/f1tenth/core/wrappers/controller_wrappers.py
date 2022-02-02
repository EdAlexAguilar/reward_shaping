import gym


class IncrementalActionWrapper(gym.Wrapper):
    """ The action is defined as increment of the previous action, ie. action += action' """

    def __init__(self, env, actionnames, increments):
        super(IncrementalActionWrapper, self).__init__(env)
        self._actionnames = actionnames
        self._increments = increments
        # update action space
        self._acts_lows = {a: self.action_space[a].low for a in self._actionnames}
        self._acts_highs = {a: self.action_space[a].high for a in self._actionnames}
        # define new normalized action space
        action_dict = {}
        for a, space in self.action_space.spaces.items():
            if a in self._actionnames:
                action_dict[a] = gym.spaces.Box(low=self._increments[a][0], high=self._increments[a][1], shape=space.shape)
            else:
                action_dict[a] = space
        self.action_space = gym.spaces.Dict(action_dict)
        #
        self.last_action = None

    def reset(self, **kwargs):
        obs = super(IncrementalActionWrapper, self).reset(**kwargs)
        self.last_action = {a: 0.0 for a in self._actionnames}
        return obs

    def step(self, action):
        for a in self._actionnames:
            self.last_action[a] = self.last_action[a] + action[a]
        return super(IncrementalActionWrapper, self).step(self.last_action)