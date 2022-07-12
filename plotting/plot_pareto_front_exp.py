import argparse
import json
import pathlib
import random
from typing import Dict, Any, List

import matplotlib.pyplot as plt
import numpy as np

COLORS = {
    "morl_lambda-0.00": '#377eb8',
    "morl_lambda-0.25": '#4daf4a',
    "morl_lambda-0.50": '#984ea3',
    "morl_lambda-0.75": '#a65628',
    "morl_lambda-1.00": '#ff7f00',
    "hprs": '#e41a1c',
}


def main(args):
    for file in args.files:
        # load data
        with open(file, "r") as f:
            data = json.load(f)
        # iterate over env, tasks
        # assume: input file is a dict env -> task -> reward -> metrics
        for env, env_data in data.items():
            for task, task_data in env_data.items():
                plot_task_frontier(task_data)


def plot_task_frontier(data: Dict[str, Dict[str, List[Any]]]):
    # prepare plot
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    # assume: data: reward -> {metric: score_per_episode}
    xs, ys, zs, labels, markers, colors = [], [], [], [], [], []
    meanxs, meanys, meanzs, meanlabels, meanmarkers, meancolors = [], [], [], [], [], []
    for reward in ["morl_lambda-0.00", "morl_lambda-0.25", "morl_lambda-0.50", "morl_lambda-0.75", "morl_lambda-1.00",
                   "hprs"]:
        if reward not in data.keys():
            continue
        all_data = data[reward]
        # get points
        safety_scores = compute_safety_score(all_data, return_mean=False)
        target_score = compute_target_score(all_data, return_mean=False)
        comfort_score = compute_comfort_score(all_data, return_mean=False)
        xs.append(safety_scores)
        ys.append(target_score)
        zs.append(comfort_score)
        labels.append(reward)
        markers.append('.')
        colors.append(COLORS[reward])
        # get means
        safety_scores = compute_safety_score(all_data, return_mean=True)
        target_score = compute_target_score(all_data, return_mean=True)
        comfort_score = compute_comfort_score(all_data, return_mean=True)
        meanxs.append(safety_scores)
        meanys.append(target_score)
        meanzs.append(comfort_score)
        meanlabels.append(reward)
        meanmarkers.append('*' if reward=="hprs" else 'x')
        meancolors.append(COLORS[reward])
    # plot
    for x, y, z, l, m, c in zip(xs, ys, zs, labels, markers, colors):
        ax.scatter(x, y, z, label=l, marker=m, s=50, c=c, alpha=0.5)
    for x, y, z, l, m, c in zip(meanxs, meanys, meanzs, meanlabels, meanmarkers, meancolors):
        ax.scatter(x, y, z, label=l, marker=m, s=100, c=c, alpha=1.0)
    # labels
    ax.set_xlabel(r"Safety: $\frac{1}{T} \sum_t s_t \models p_{safety}$")
    ax.set_ylabel(r"Target: $\frac{\rho(s_T, p_{target}) - \rho_{min}}{\rho_{max} - \rho_{min}} $")
    ax.set_zlabel(r"Comfort: $\frac{1}{T} \sum_t 1/C \sum_c s_t \models p_{comfort_c}$")
    # axis lengths
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    # plot
    ax.legend()
    plt.show()


def compute_safety_score(data, return_mean: bool = True) -> float:
    # assume: data: metric -> list of score per episode
    # idea: use the normalized counter of safety reqs as metric for safety requirements
    metric_names = [k for k in data.keys() if k.startswith("s") and "counter" in k]
    norm_safety_scores = [1 / 401 * np.array(data[m]) for m in metric_names]  # norm w.r.t. episode len
    norm_safety_scores = np.min(norm_safety_scores, axis=0)  # average over comfort reqs
    if return_mean:
        return np.mean(norm_safety_scores)
    return norm_safety_scores


def compute_target_score(data, return_mean: bool = True) -> float:
    # assume: data: metric -> list of score per episode
    # idea: use the robustness of the last timestep (stored as t_%s_lastrob) as metric for target requirement
    metric_name = [k for k in data.keys() if k.startswith("t_") and "lastrob" in k]
    assert len(metric_name) == 1, "more than 1 metric found for target req"
    metric_name = metric_name[0]
    mean = np.array(data[metric_name])
    # normalize it in 0 - 1
    minr, maxr = -3.7, +0.25
    norm_target_score = (mean - minr) / (maxr - minr)
    if return_mean:
        return np.mean(norm_target_score)
    return norm_target_score


def compute_comfort_score(data, return_mean: bool = True) -> float:
    # assume: data: metric -> list of score per episode
    # idea: use the normalized counter of comfort reqs as metric for comfort requirements
    metric_names = [k for k in data.keys() if k.startswith("c") and "counter" in k]
    norm_cscores = [1 / 401 * np.array(data[m]) for m in metric_names]  # norm w.r.t. episode len
    norm_cscores = np.mean(norm_cscores, axis=0)  # average over comfort reqs
    if return_mean:
        return np.mean(norm_cscores)
    return norm_cscores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", type=pathlib.Path, required=True, nargs='+')
    parser.add_argument("-save", action="store_true")
    parser.add_argument("-legend", action="store_true")
    parser.add_argument("-info", action="store_true")
    args = parser.parse_args()
    main(args)
