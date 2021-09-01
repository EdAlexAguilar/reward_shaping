import argparse
import json
import pathlib
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt

REQ_LABELS = {'no_collision': 'AvoidCollision',
              'no_falldown': 'AvoidFalldown',
              'no_outside': 'AvoidExit',
              'reach_origin': 'ReachTarget',
              'balance': 'ReachBalancing'
              }


def plot_barplot(result_dict, ylabel):
    plt.bar(range(len(result_dict.keys())), result_dict.values())
    plt.ylabel(ylabel)
    plt.xticks(range(len(result_dict.keys())), result_dict.keys())


def plot_results(results: Dict[str, List[float]]):
    # evaluate
    mean_rob, sat_rate = {}, {}
    n_instances = 0
    for k, rr in results.items():
        n_instances = len(rr)
        mean_rob[k] = np.mean(rr)
        sat_rate[k] = np.mean([float(r >= 0) for r in rr])
    # plot mean rob
    plt.subplot(1, 2, 1)
    plot_barplot(mean_rob, "Mean Robustness")
    plt.subplot(1, 2, 2)
    plot_barplot(sat_rate, "Satisfaction Rate")
    plt.show()
    print(f"[Info] Results average over {n_instances} instance")


def main(args):
    with open(args.file, 'r') as f:
        data = json.load(f)
    if args.no_aggregate:
        results = [{REQ_LABELS[k]: [instance[k]] for k in instance.keys()} for instance in data]
    else:
        results = [{REQ_LABELS[k]: [res[k] for res in data] for k in data[0].keys()}]
    for result in results:
        plot_results(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=pathlib.Path, required=True)
    parser.add_argument("-no_aggregate", action='store_true')
    # parse args
    args = parser.parse_args()
    main(args)
