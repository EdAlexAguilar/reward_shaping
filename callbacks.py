import logging
import sys
from typing import Any, Dict

import gym
import torch as th

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import Video

import rtamt
import numpy as np

class VideoRecorderCallback(BaseCallback):
    def __init__(self, eval_env: gym.Env, render_freq: int, n_eval_episodes: int = 1, deterministic: bool = True):
        """
        Records a video of an agent's trajectory traversing ``eval_env`` and logs it to TensorBoard

        :param eval_env: A gym environment from which the trajectory is recorded
        :param render_freq: Render the agent's trajectory every eval_freq call of the callback.
        :param n_eval_episodes: Number of episodes to render
        :param deterministic: Whether to use deterministic or stochastic policy
        """
        super().__init__()
        self._eval_env = eval_env
        self._render_freq = render_freq
        self._n_eval_episodes = n_eval_episodes
        self._deterministic = deterministic

    def _on_step(self) -> bool:
        if self.n_calls % self._render_freq == 0:
            screens = []

            def grab_screens(_locals: Dict[str, Any], _globals: Dict[str, Any]) -> None:
                """
                Renders the environment in its current state, recording the screen in the captured `screens` list

                :param _locals: A dictionary containing all local variables of the callback's scope
                :param _globals: A dictionary containing all global variables of the callback's scope
                """
                screen = self._eval_env.render(mode="rgb_array")
                # PyTorch uses CxHxW vs HxWxC gym (and tensorflow) image convention
                screens.append(screen.transpose(2, 0, 1))

            evaluate_policy(
                self.model,
                self._eval_env,
                callback=grab_screens,
                n_eval_episodes=self._n_eval_episodes,
                deterministic=self._deterministic,
            )
            self.logger.record(
                "trajectory/video",
                Video(th.ByteTensor([screens]), fps=40),
                exclude=("stdout", "log", "json", "csv"),
            )
        return True


class RobustMonitoringCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, env_params, every_n_rollouts=1000, verbose=0):
        super(RobustMonitoringCallback, self).__init__(verbose)
        self.logging = False
        self.every_n_rollouts = every_n_rollouts
        self.rollout_k = 0
        # spec
        self.variables = ['time', 'collision', 'falldown', 'outside', 'dist_target_x', 'dist_obstacle', 'dist_target_theta']
        self.types = ['int', 'int', 'int', 'int', 'float', 'float', 'float']
        safety_requirements = "always((collision>=0) and (outside>=0) and not(falldown>=0))"
        target_requirements = f"eventually(dist_target_x <= {env_params['x_target_tol']})"
        balancing_requirement = f"always((dist_obstacle >= 0.5) -> (dist_target_theta <= {env_params['theta_target_tol']}))"
        self.spec = f"{safety_requirements} and {target_requirements} and {balancing_requirement}"
        # rollout
        self.rollout = {v: [] for v in self.variables}

    def _on_step(self) -> bool:
        if not self.logging:
            return True
        env = self.training_env.envs[0]
        x, theta = env.state[0], env.state[2]
        collision = -1 * env.obstacle.intersect(x, theta)
        falldown = -1 * (theta > env.theta_threshold_radians)
        outside = -1 * (x > env.x_threshold)
        dist_target_x = abs(x - env.x_target)
        dist_target_theta = abs(theta - env.theta_target)
        dist_obstacle = abs(x - (env.obstacle.left_x + (env.obstacle.right_x - env.obstacle.left_x)/2.0))
        #
        if env.step_count == 0 and len(self.rollout['time'])>0:
            self._on_rollout_end()
            return True
        self.rollout['time'].append(env.step_count)
        self.rollout['collision'].append(collision)
        self.rollout['falldown'].append(falldown)
        self.rollout['outside'].append(outside)
        self.rollout['dist_target_x'].append(dist_target_x)
        self.rollout['dist_target_theta'].append(dist_target_theta)
        self.rollout['dist_obstacle'].append(dist_obstacle)
        return True

    def _on_rollout_start(self) -> None:
        self.rollout_k += 1
        if self.rollout_k % self.every_n_rollouts == 0:
            self.logging = True

    def _on_rollout_end(self) -> None:
        if not self.logging:
            return
        # compute robustness
        spec = rtamt.STLSpecification()
        for v, t in zip(self.variables, self.types):
            spec.declare_var(v, f'{t}')
        spec.spec = self.spec
        try:
            spec.parse()
        except rtamt.STLParseException as err:
            logging.error('STL Parse Exception: {}'.format(err))
            return
        # preprocess format, evaluate, post process
        robustness_trace = spec.evaluate(self.rollout)
        robustness = robustness_trace[0][1]
        # log on tensorboard
        self.logger.record('robustness', robustness)
        # reset
        self.rollout = {v: [] for v in self.variables}
        self.logging = False
