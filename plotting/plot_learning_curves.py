import argparse
import pathlib
import time
import warnings
from typing import Dict, List, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plotting.custom_evaluations import get_custom_evaluation


def get_files(logdir, regex):
    return logdir.glob(f"{regex}/evaluations*.npz")


def parse_env_task(filepath: str):
    env, task = None, None
    for env_name in ["cart_pole_obst", "bipedal_walker", "lunar_lander", "racecar"]:
        if env_name in filepath:
            env = env_name
            break
    for task_name in ["fixed_height", "forward", "hardcore", "land", "drive"]:
        if task_name in filepath:
            task = task_name
            break
    if not env or not task:
        raise ValueError(f"not able to parse env/task in {filepath}")
    return env, task


def get_evaluations(logdir: pathlib.Path, regex: str) -> List[Dict[str, np.ndarray]]:
    """ look for evaluations.npz in the subdirectories and return is content """
    evaluations = []
    for eval_file in get_files(logdir, regex):
        data = dict(np.load(eval_file))
        data["filepath"] = str(eval_file)
        evaluations.append(data)
    if len(evaluations) == 0:
        warnings.warn(f"cannot find any file for `{logdir}/{regex}/evaluations.npz`", UserWarning)
    return evaluations


def aggregate_evaluations(evaluations: List[Dict[str, np.ndarray]], params: Dict) -> Dict[str, np.ndarray]:
    """ aggregate data in the list to produce overall figure """
    # make flat collection of x,y data
    xx, yy = [], []
    for evaluation in evaluations:
        assert params['x'] in evaluation, f"{params['x']} is not in evaluation keys ({evaluation.keys()})"
        assert params['y'] in evaluation, f"{params['y']} is not in evaluation keys ({evaluation.keys()})"
        xx.append(evaluation[params['x']])
        yy.append(np.mean(evaluation[params['y']], axis=-1))  # eg, average results over multiple eval episodes
    xx = np.concatenate(xx, axis=0)
    yy = np.concatenate(yy, axis=0)
    assert xx.shape == yy.shape, f"xx, yy dimensions don't match (xx shape:{xx.shape}, yy shape:{yy.shape}"
    # aggregate
    df = pd.DataFrame({'x': xx, 'y': yy})
    bins = np.arange(0, max(xx) + params['binning'], params['binning'])
    df['xbin'] = pd.cut(df['x'], bins=bins, labels=bins[:-1])
    aggregated = df.groupby("xbin").agg(['mean', 'std'])['y'].reset_index()
    return {'x': aggregated['xbin'].values,
            'mean': aggregated['mean'].values,
            'std': aggregated['std'].values}


def plot_data(data: Dict[str, np.ndarray], ax: plt.Axes, **kwargs):
    assert all([key in data for key in ['x', 'mean', 'std']]), f'x, mean, std not found in data (keys: {data.keys()})'
    ax.plot(data['x'], data['mean'], **kwargs)
    ax.fill_between(data['x'], data['mean'] - data['std'], data['mean'] + data['std'], alpha=0.5)
    ax.set_ylim(0.0, 2.0)


def plot_file_info(args):
    for regex in args.regex:
        files = [f for f in get_files(args.logdir, regex)]
        print(f"regex: {regex}, nr files: {len(files)}")
        for f in files:
            data = dict(np.load(f))
            keys = [k for k in data.keys()]
            print(f"  file: {f.stem}, keys: {keys}")


def extend_with_custom_evaluation(evaluations, y):
    custom_fn = get_custom_evaluation(y)
    for i in range(len(evaluations)):
        if y in evaluations[i]:
            continue
        env_name, task_name = parse_env_task(evaluations[i]["filepath"])
        evaluations[i][y] = custom_fn(data=evaluations[i], env_name=env_name)
    return evaluations


def main(args):
    # only print info on files
    if args.info:
        plot_file_info(args)
        exit(0)
    # prepare plot
    fig, ax = plt.subplots(nrows=1, ncols=1)
    # plot data
    for regex in args.regex:
        evaluations = get_evaluations(args.logdir, regex)
        if not evaluations:
            continue
        if any([args.y not in evaluation.keys() for evaluation in evaluations]):
            evaluations = extend_with_custom_evaluation(evaluations, args.y)
        data = aggregate_evaluations(evaluations, params={'x': args.x, 'y': args.y, 'binning': args.binning})
        plot_data(data, ax, label=regex)
    plt.legend()
    if args.save:
        plot_dir = args.logdir / "plots"
        plot_dir.mkdir(exist_ok=True, parents=True)
        outfile = plot_dir / f"plot_{int(time.time())}.pdf"
        plt.savefig(outfile)
        print(f"[Info] Figure saved in {outfile}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=pathlib.Path, required=True)
    parser.add_argument("--regex", type=str, default="**", nargs="+",
                        help="for each regex, group data for `{logdir}/{regex}/evaluations*.npz`")
    parser.add_argument("--binning", type=int, default=15000)
    parser.add_argument("--x", type=str, default="timesteps")
    parser.add_argument("--y", type=str, default="results")
    parser.add_argument("-save", action="store_true")
    parser.add_argument("-info", action="store_true")
    args = parser.parse_args()
    main(args)
