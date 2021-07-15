from typing import Dict, List, Tuple

import numpy as np
import gym

from envs.cart_pole.cp_continuous_env import CartPoleContEnv
from hierarchy.graph import HierarchicalGraph


class HierarchicalGraphRewardWrapper(gym.RewardWrapper):
    """
    Hierarchical Reward Definition, using DAG hierarchy.
    """

    def __init__(self, env, hierarchy: HierarchicalGraph, use_potential=False):
        """
        TODO
        """
        super().__init__(env)
        self.hierarchy = hierarchy
        # aux variables, for rendering
        self.rew = 0.0
        self.ret = 0.0
        self.use_potential = use_potential
        self.last_state = None

    def reset(self, **kwargs):
        state = self.env.reset(**kwargs)
        self.last_state = state
        self.rew = 0.0
        self.ret = 0.0
        return state

    def reward(self, rew):
        # note: the need of this overwriting fo rew/ret is purely for rendering purposes
        # in this way, the env.render method with render the correct reward
        if self.use_potential and self.env.step_count > 0:
            # assume gamma == 1
            self.rew = rew + self.reward_in_state(self.state) - self.reward_in_state(self.last_state)
        else:
            self.rew = self.reward_in_state(self.state)
        self.ret += self.rew
        self.env.rew = self.rew
        self.env.ret = self.ret
        return self.rew

    def reward_in_state(self, state):
        """
        TODO
        """
        if state is None:
            raise ValueError("eval reward in not initialized env, state is None")
        scores = np.zeros(len(self.hierarchy.labels))   # all zeros
        enable = np.ones(len(self.hierarchy.labels)).astype(bool)  # all true
        for node_label in self.hierarchy.top_sorting:
            node_id = self.hierarchy.lab2id[node_label]
            scores[node_id] = self.hierarchy.score[node_id](state)
            assert scores[node_id] >= 0.0 and scores[node_id] <= 1.0
            node_val = self.hierarchy.valuation[node_id](state)
            neighbours_ids = [self.hierarchy.lab2id[neigh] for neigh in self.hierarchy.neighbors(node_label)]
            for neigh_id in neighbours_ids:
                enable[neigh_id] = enable[neigh_id] and node_val
        final_reward = np.sum(scores * enable)
        return final_reward


def prova():
    expected_result = True
    try:
        labels = ["S1", "S2", "S3", "T1", "C1"]
        f = lambda x: 1.0
        v = lambda x: False
        scores = [f] * len(labels)
        values = [v] * len(labels)
        edges = [("S1", "T1"), ("S2", "T1"), ("S3", "T1"), ("T1", "C1")]
        g = HierarchicalGraph(labels, scores, values, edges)
        env = CartPoleContEnv(task='balance')
        env = HierarchicalGraphRewardWrapper(env, g)
        # evaluation
        obs = env.reset()
        env.render()
        rewards = []
        tot_reward = 0.0
        for i in range(100):
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            tot_reward += reward
            env.render()
            if done:
                rewards.append(tot_reward)
                obs = env.reset()
                rob = env.compute_episode_robustness(env.last_complete_episode)
                print(f"reward: {tot_reward:.3f}, robustness: {rob:.3f}")
                tot_reward = 0.0
            input()
        result = True
    except Exception as e:
        print(e)
        result = False
    return result == expected_result

if __name__=="__main__":
    print(prova())