# assumption: all the files in the logdir referring the same reward have to be aggregated
import argparse
import pathlib
import time
import warnings
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PALETTE = ['#377eb8', '#4daf4a', '#984ea3', '#e41a1c', '#ff7f00', '#a65628', '#888888', '#fdbf6f']
REWARDS = {'stl': 'STL', 'weighted': 'Weighted',
           'gb_chain': 'GB-Chain', 'gb_bcr_bi': 'GBH-DistTarget-BinarySat', 'gb_pcr_bi': 'GBH-ProgressTarget-BinarySat',
           'gb_bpr_ci': 'GBH-ProgressTarget-ContinuousSat',
           'continuous': 'Continuous'}
PALETTE_REWARDS = {reward: v for reward, v in zip(REWARDS.keys(), PALETTE)}

PALETTE_HLINES = {-1: 'red', 0: 'green'}

STYLE = {'fill_alpha': 0.2, 'hline_alpha': 0.5, 'hline_style': 'dashed'}


def extract_reward(path):
    for reward in REWARDS.keys():
        if reward in path.stem:
            return reward
    warnings.warn(f"skipped: not able to extract reward from {str(path)}")


def load_data(rewards: List[str], inpath: pathlib.Path, regex: str, tag: str) -> Dict:
    # load data and create a dict with x (steps) and y (tag)
    data = {reward: {'x': [], 'y': []} for reward in rewards}
    for filepath in inpath.glob(regex):
        if not filepath.parts[-1].endswith("csv"):
            print(f"Skip non-csv: {filepath}")
            continue
        reward = extract_reward(filepath)
        if reward not in rewards or reward is None:
            continue
        tdf = pd.read_csv(filepath, index_col=[0, 1])
        tdf = tdf.loc[tag, "value"]
        x, y = tdf.index.astype(int).to_numpy(), tdf.values
        data[reward]['x'] = np.concatenate([data[reward]['x'], x]).astype(int)
        data[reward]['y'] = np.concatenate([data[reward]['y'], y])
    return data


def aggregate_mean_std(data, binning: int):
    # create binned data (steps are binned), and aggregate y to compute mean and std
    df = pd.DataFrame(data, columns=data.keys())
    bins = np.arange(0, max(data['x']) + binning, binning)
    df['xbin'] = pd.cut(df['x'], bins=bins, labels=bins[:-1])
    aggregated = df.groupby("xbin").agg(['mean', 'std'])['y']
    return aggregated


def plot_line(reward, df):
    assert 'mean' in df.columns and 'std' in df.columns
    # data
    x, y_mean = df.index, df['mean']
    y_plus_std, y_minus_std = df['mean'] + df['std'], df['mean'] - df['std']
    # style
    color = PALETTE_REWARDS[reward]
    plt.plot(x, y_mean, color=color, label=REWARDS[reward])
    plt.fill_between(x, y_plus_std, y_minus_std, color=color, alpha=STYLE['fill_alpha'])


def plot_secondaries(xlabel, ylabel, hlines, minx, maxx):
    # draw horizonal lines
    for value in hlines:
        plt.hlines(value, minx, maxx,
                   color=PALETTE_HLINES[value], alpha=STYLE['hline_alpha'], linestyles=STYLE['hline_style'])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()


def main(args):
    all_data = load_data(args.rewards, args.path, args.regex, args.tag)
    minx, maxx = np.Inf, 0
    for reward, data in all_data.items():
        if len(data['x']) == 0:
            continue
        minx = min(minx, min(data['x']))
        maxx = max(maxx, max(data['x']))
        aggregated = aggregate_mean_std(data, binning=args.binning)
        plot_line(reward, aggregated)
    plot_secondaries(xlabel=args.xlabel, ylabel=args.ylabel, hlines=args.hlines, minx=minx, maxx=maxx)
    if args.save:
        outfile = args.path / f'learning_curves_{int(time.time())}.pdf'
        plt.savefig(outfile)
        print(f"\n\nPlot exported in {outfile}")

    else:
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=pathlib.Path, required=True, help="input dir where look for exports in csv")
    parser.add_argument("--regex", type=str, default="*", help="arbitrary regex to filter selection")
    parser.add_argument("--tag", type=str, default="eval/mean_reward", help="tag plot on y axis")
    parser.add_argument("--binning", type=int, default=15000, help="binning to aggregate data")
    parser.add_argument("--xlabel", type=str, default='Steps', help="label x axis")
    parser.add_argument("--ylabel", type=str, default='Y', help="label y axis")
    parser.add_argument("--hlines", type=float, nargs='*', default=[0, -1], help="horizontal lines in plot, eg. y=0")
    parser.add_argument("--rewards", type=str, nargs='*', default=REWARDS.keys(), choices=REWARDS.keys(),
                        help="rewards to be plotted")
    parser.add_argument("-save", action='store_true')
    args = parser.parse_args()

    inpath = args.path
    main(args)