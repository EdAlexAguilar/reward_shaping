import pathlib
from typing import Any, Dict, Union, Optional

import gym
import torch as th
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import Video

import numpy as np
import os

from stable_baselines3.common.vec_env import sync_envs_normalization, VecEnv

from reward_shaping.training.custom_evaluation import evaluate_policy_with_monitors


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

            mean_reward, std_reward = evaluate_policy(
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


class CustomEvalCallback(EvalCallback):

    def __init__(
            self,
            eval_env: Union[gym.Env, VecEnv],
            callback_on_new_best: Optional[BaseCallback] = None,
            callback_after_eval: Optional[BaseCallback] = None,
            n_eval_episodes: int = 5,
            eval_freq: int = 10000,
            log_path: Optional[str] = None,
            best_model_save_path: Optional[str] = None,
            deterministic: bool = True,
            render: bool = False,
            verbose: int = 1,
            warn: bool = True,
    ):
        super(CustomEvalCallback, self).__init__(eval_env, callback_on_new_best, callback_after_eval,
                                                 n_eval_episodes, eval_freq, log_path,
                                                 best_model_save_path, deterministic, render, verbose, warn)
        # initialize list of metrics, in our case the metric of an individual requirement (e.g., s1_nofalldown_counter)
        self._list_of_metrics = [f"{req}_counter" for req in eval_env.req_labels]
        self.evaluations_metrics = {m: [] for m in self._list_of_metrics}
        # for logging
        self.log_dir = pathlib.Path(log_path)

    def _on_step(self) -> bool:

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sync training and eval env if there is VecNormalize
            sync_envs_normalization(self.training_env, self.eval_env)

            # Reset success rate buffer
            self._is_success_buffer = []

            episode_rewards, episode_lengths, episode_metrics = evaluate_policy_with_monitors(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._log_success_callback,
                list_of_metrics=self._list_of_metrics
            )

            if self.log_path is not None:
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)
                # update metrics
                for m, v in episode_metrics.items():
                    self.evaluations_metrics[m].append(episode_metrics[m])

                kwargs = {}
                # Save success log if present
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    kwargs = dict(successes=self.evaluations_successes)

                # note: EvalCallback set logpath to `logdir/evaluations`
                # we prefer `logdir/evaluations_<something_identifying_experiment>` to easier post-processing
                exp_identifier = self.log_dir.name
                log_path = self.log_dir / f"evaluations_{exp_identifier}"
                np.savez(
                    str(log_path),
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    **self.evaluations_metrics,     # unpack eval metrics to
                    **kwargs,
                )

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_metrics = {m: np.mean(episode_metrics[m]) for m in episode_metrics}
            std_metrics = {m: np.std(episode_metrics[m]) for m in episode_metrics}
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            self.last_mean_reward = mean_reward

            if self.verbose > 0:
                print(
                    f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                for m in episode_metrics:
                    print(f"Eval {m}={mean_metrics[m]:.2f} +/- {std_metrics[m]:.2f}")
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
            # Add to current Logger
            self.logger.record("eval/mean_reward", float(mean_reward))
            self.logger.record("eval/mean_ep_length", mean_ep_length)
            for m in mean_metrics:
                self.logger.record(f"reqs/{m}", float(mean_metrics[m]))

            if len(self._is_success_buffer) > 0:
                success_rate = np.mean(self._is_success_buffer)
                if self.verbose > 0:
                    print(f"Success rate: {100 * success_rate:.2f}%")
                self.logger.record("eval/success_rate", success_rate)

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record("time/total timesteps", self.num_timesteps, exclude="tensorboard")
            self.logger.dump(self.num_timesteps)

            if mean_reward > self.best_mean_reward:
                if self.verbose > 0:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                self.best_mean_reward = mean_reward
                # Trigger callback if needed
                if self.callback is not None:
                    return self._on_event()

        return True
