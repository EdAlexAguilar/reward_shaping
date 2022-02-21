import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import gym
import numpy as np

from stable_baselines3.common import base_class
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecMonitor, is_vecenv_wrapped


def evaluate_policy_with_monitors(
        model: "base_class.BaseAlgorithm",
        env: Union[gym.Env, VecEnv],
        n_eval_episodes: int = 10,
        deterministic: bool = True,
        render: bool = False,
        callback: Optional[Callable[[Dict[str, Any], Dict[str, Any]], None]] = None,
        reward_threshold: Optional[float] = None,
        return_episode_rewards: bool = False,
        warn: bool = True,
        list_of_metrics: List[str] = None,
) :
    """
    This function extends the original stable_baselines3.common.evaluation.evaluate_policy
    to also return a dictionary of metrics for logging purposes.

    .. note::
        The original idea is to log the results from the monitors, that are available in the last step of the episode.
        For this reason, we assume that:
        - the metrics listed in `list_of_metrics` are stored in the `ìnfo` dictionary,
        - they are meaningful only in the last step.

    @param list_of_metrics: list of metrics name
    """
    is_monitor_wrapped = False
    # Avoid circular import
    from stable_baselines3.common.monitor import Monitor

    if not isinstance(env, VecEnv):
        env = DummyVecEnv([lambda: env])

    is_monitor_wrapped = is_vecenv_wrapped(env, VecMonitor) or env.env_is_wrapped(Monitor)[0]

    if not is_monitor_wrapped and warn:
        warnings.warn(
            "Evaluation environment is not wrapped with a ``Monitor`` wrapper. "
            "This may result in reporting modified episode lengths and rewards, if other wrappers happen to modify these. "
            "Consider wrapping environment first with ``Monitor`` wrapper.",
            UserWarning,
        )

    n_envs = env.num_envs
    episode_rewards = []
    episode_lengths = []
    episode_metrics = {metric: [] for metric in list_of_metrics}

    episode_counts = np.zeros(n_envs, dtype="int")
    # Divides episodes among different sub environments in the vector as evenly as possible
    episode_count_targets = np.array([(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype="int")

    current_rewards = np.zeros(n_envs)
    current_lengths = np.zeros(n_envs, dtype="int")
    observations = env.reset()
    states = None
    while (episode_counts < episode_count_targets).any():
        actions, states = model.predict(observations, state=states, deterministic=deterministic)
        observations, rewards, dones, infos = env.step(actions)
        current_rewards += rewards
        current_lengths += 1
        for i in range(n_envs):
            if episode_counts[i] < episode_count_targets[i]:

                # unpack values so that the callback can access the local variables
                reward = rewards[i]
                done = dones[i]
                info = infos[i]

                if callback is not None:
                    callback(locals(), globals())

                if dones[i]:
                    # Note: in some env (e.g., Atari), the flag done is true when the agent looses a life
                    # and this does not correspond to the real end of the episode.
                    # In our case studies, the done flag is true only at the episode end,
                    # so we do not consider other scenarios.
                    episode_rewards.append(current_rewards[i])
                    episode_lengths.append(current_lengths[i])
                    for m in list_of_metrics:
                        assert m in episode_metrics, f"{m} not found in list of metrics"
                        assert m in info, f"{m} not found in info dictionary"
                        episode_metrics[m].append(info[m])
                    episode_counts[i] += 1
                    current_rewards[i] = 0
                    current_lengths[i] = 0
                    if states is not None:
                        states[i] *= 0

        if render:
            env.render()

    mean_length = np.mean(episode_lengths)
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_metrics = {m: np.mean(episode_metrics[m]) for m in list_of_metrics}
    std_metrics = {m: np.std(episode_metrics[m]) for m in list_of_metrics}
    if reward_threshold is not None:
        assert mean_reward > reward_threshold, "Mean reward below threshold: " f"{mean_reward:.2f} < {reward_threshold:.2f}"
    if return_episode_rewards:
        return episode_rewards, episode_lengths, episode_metrics
    return mean_reward, std_reward, mean_metrics, std_metrics
