import gym


class STLReward(gym.RewardWrapper):
    """
    reward(s,a) := rho(phi,episode), if terminal
    reward(s,a) := 0, otherwise

    note: this reward in non-markovian because requires to observe the whole episode
    """

    def __init__(self, env):
        super().__init__(env)
        self.rew = 0.0
        self.ret = 0.0

    def reward(self, reward):
        # note: the need of this overwriting fo rew/ret is purely for rendering purposes
        # in this way, the env.render method with render the correct reward
        self.rew = self.reward_in_state(self.state)
        self.ret += self.rew
        self.env.rew = self.rew
        self.env.ret = self.ret
        return self.rew

    def reward_in_state(self, state):
        if self.env.done:
            return self.env.compute_episode_robustness(self.env.last_complete_episode,
                                                       self.env.last_cont_spec)
        else:
            return 0.0

    def reset(self):
        obs = super(STLReward, self).reset()
        self.rew = 0.0
        self.ret = 0.0
        return obs


class BoolSTLReward(gym.RewardWrapper):
    """
    reward(s,a) := rho(phi,episode), if terminal
    reward(s,a) := 0, otherwise

    note: this reward in non-markovian because requires to observe the whole episode
    """

    def __init__(self, env):
        super().__init__(env)
        self.rew = 0.0
        self.ret = 0.0

    def reward(self, reward):
        # note: the need of this overwriting fo rew/ret is purely for rendering purposes
        # in this way, the env.render method with render the correct reward
        self.rew = self.reward_in_state(self.state)
        self.ret += self.rew
        self.env.rew = self.rew
        self.env.ret = self.ret
        return self.rew

    def reward_in_state(self, state):
        if self.env.done:
            return self.env.compute_episode_robustness(self.env.last_complete_episode,
                                                       self.env.last_bool_spec)
        else:
            return 0.0

    def reset(self):
        obs = super(BoolSTLReward, self).reset()
        self.rew = 0.0
        self.ret = 0.0
        return obs


class SparseReward(gym.RewardWrapper):
    """
    reward is sparsely assigned (e.g. when reaching the target or when failing)
    """
    def __init__(self, env):
        super().__init__(env)
        self.rew = 0.0
        self.ret = 0.0

    def reward(self, reward):
        # note: the need of this overwriting fo rew/ret is purely for rendering purposes
        # in this way, the env.render method with render the correct reward
        self.rew = self.reward_in_state(self.state)
        self.ret += self.rew
        self.env.rew = self.rew
        self.env.ret = self.ret
        return self.rew

    def reward_in_state(self, state):
        if self.env.task == "balance":
            reward = 1.0 if abs(state[2]) <= self.env.theta_threshold_radians else 0.0
        elif self.env.task == "target":
            reward = 1.0 if abs(state[2]) <= self.env.theta_threshold_radians else 0.0
            reward += +10 if abs(state[0] - self.env.x_target) <= self.env.x_target_tol else 0.0
        else:
            raise NotImplemented(f"no reward for task {self.env.task}")
        return reward

    def reset(self):
        obs = super(SparseReward, self).reset()
        self.rew = 0.0
        self.ret = 0.0
        return obs


class ContinuousReward(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.rew = 0.0
        self.ret = 0.0

    def reward(self, reward):
        # note: the need of this overwriting fo rew/ret is purely for rendering purposes
        # in this way, the env.render method with render the correct reward
        self.rew = self.reward_in_state(self.state)
        self.ret += self.rew
        self.env.rew = self.rew
        self.env.ret = self.ret
        return self.rew

    def reward_in_state(self, state):
        if self.env.task == "balance":
            norm_angle = abs(self.env.state[2]) / self.env.theta_threshold_radians
            reward = (1 - norm_angle)
        elif self.env.task == "target":
            norm_angle = abs(self.env.state[2]) / self.env.theta_threshold_radians
            dist_target = abs(state[0]) / self.env.x_threshold
            reward = (1 - norm_angle) ** 2 + (1 - dist_target) ** 2
        else:
            raise NotImplemented(f"no reward for task {self.env.task}")
        return reward

    def reset(self):
        obs = super(ContinuousReward, self).reset()
        self.rew = 0.0
        self.ret = 0.0
        return obs
