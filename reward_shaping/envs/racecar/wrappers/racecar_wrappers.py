import gym

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
