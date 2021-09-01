import argparse
import json
import pathlib
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

NODE_LABELS = {}


def plot_barplot(result_dict, title):
    plt.bar(range(len(result_dict.keys())), result_dict.values())
    plt.title(title)
    plt.axhline(0, color='k', linewidth=0.8)
    lowy = -1.1 if title == "Score" else 0.0
    plt.ylim(lowy, 1.1)
    plt.xticks(range(len(result_dict.keys())), result_dict.keys())


def plot_results(results: Dict[str, Tuple[float, float, float]]):
    # evaluate
    rewards, sats, scores = {}, {}, {}
    for k, (reward, sat, score) in results.items():
        rewards[k] = reward
        sats[k] = sat
        scores[k] = score
    # plot mean rob
    plt.figure(figsize=(9, 5))
    plt.tight_layout()
    plt.axis(True)
    plt.subplot(1, 3, 1)
    plot_barplot(rewards, "Reward")
    plt.subplot(1, 3, 2)
    plot_barplot(sats, "Sat Degree")
    plt.subplot(1, 3, 3)
    plot_barplot(scores, "Score")
    plt.show()


def main(args):
    with open(args.file, 'r') as f:
        results = json.load(f)
    for result in results:
        plot_results(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=pathlib.Path, required=True)
    # parse args
    args = parser.parse_args()
    main(args)
