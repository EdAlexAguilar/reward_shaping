import argparse
import math
import pathlib
from typing import List

import matplotlib.pyplot as plt
import time

import numpy as np
from matplotlib import animation
from matplotlib.animation import FuncAnimation

limits = {
    "horizontal_speed": [0.30, 1.0],
    "vertical_speed": [-0.1, 0.1],
    "hull_angle": [-0.08726, 0.08726],
    "hull_angle_speed": [-0.25, 0.25]
}
labels = {
    "horizontal_speed": "Horizontal Velocity",
    "vertical_speed": "Vertical Velocity",
    "hull_angle": "Hull Angle",
    "hull_angle_speed": "Hull Angular Velocity",
}
show_labels = ["horizontal_speed", "hull_angle", "hull_angle_speed"]

MARGIN = 0.25  # percentage
BACKWARD_HISTORY = 0.30
FORWARD_HISTORY = 0.10

COLORS = ['#e41a1c', '#4daf4a', '#377eb8', '#984ea3', '#a65628', ]
LINEWIDTH = 5.0

FIGSIZE = (20, 5)
LARGESIZE, MEDIUMSIZE, SMALLSIZE = 25, 20, 15

plt.rcParams.update({'font.size': MEDIUMSIZE})
plt.rcParams.update({'axes.titlesize': LARGESIZE})
plt.rcParams.update({'axes.labelsize': LARGESIZE})
plt.rcParams.update({'xtick.labelsize': MEDIUMSIZE})
plt.rcParams.update({'ytick.labelsize': MEDIUMSIZE})
plt.rcParams.update({'legend.fontsize': MEDIUMSIZE})
plt.rcParams.update({'figure.titlesize': MEDIUMSIZE})


def _convert_array_to_dict(state):
    vars = "collision,ground_contact_leg0,ground_contact_leg1,horizontal_speed,hull_angle,hull_angle_speed," \
           "joint0_angle,joint0_angle_speed,joint1_angle,joint1_angle_speed,joint2_angle,joint2_angle_speed," \
           "joint3_angle,joint3_angle_speed,lidar,vertical_speed"
    dictionary = {v: s for v, s in zip(vars.split(","), state)}
    dictionary["lidar"] = state[len(vars.split(",")):]
    return dictionary


def parse_curve_name(filename: str):
    if "hrs_pot_nocomf" in filename:
        return "-Comfort"
    elif "hrs_pot" in filename:
        return "+Comfort"
    return "Unknown"


def produce_animation(trace: np.ndarray, curve: str, var: str, color="k", save: bool = False, outfile="test.gif"):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    # prepare data
    x, yy = [], []
    values = [_convert_array_to_dict(obs)[var] for obs in trace]
    # animate
    n_frames = len(values)
    backward_margin = int(BACKWARD_HISTORY * n_frames)
    forward_margin = int(FORWARD_HISTORY * n_frames)
    margin = (limits[var][1] - limits[var][0]) * MARGIN

    def animate(t):
        ax.clear()
        if t < len(values):
            x.append(t)
            yy.append(values[t])
            ax.plot(x, yy, label=curve, color=color, linewidth=LINEWIDTH)
        ax.hlines(limits[var][0], xmin=0, xmax=1000, linewidth=LINEWIDTH)
        ax.hlines(limits[var][1], xmin=0, xmax=1000, linewidth=LINEWIDTH)
        ax.set_xlim([max(0, t - backward_margin), min(t + forward_margin, n_frames)])
        ax.set_ylim([limits[var][0] - margin, limits[var][1] + margin])
        #ax.set_xlabel("Step", horizontalalignment='right', x=1.0)
        #ax.set_ylabel(labels[var])

    anim = FuncAnimation(fig, animate, frames=n_frames, interval=10, repeat=False)
    if save:
        file = pathlib.Path(outfile)
        writergif = animation.PillowWriter(fps=100)
        anim.save(file, writer=writergif)
    else:
        plt.show()


def main(args):
    traces = []  # data
    curves = args.curves  # name of the curve
    for f in args.logfiles:
        data = np.load(str(f))
        traces.append(data["observations"])
    assert len(curves) == len(args.logfiles), "nr logfile != nr curve names"

    for i, (trace, curve) in enumerate(zip(traces, curves)):
        print(curve)
        for j, var in enumerate(show_labels):
            print(f"\t{var}")
            outfile = f"comfort_plot_{curve}_{labels[var]}_{int(time.time())}.gif"
            color = COLORS[i]
            produce_animation(trace, curve, var, color=color, save=args.save, outfile=outfile)


if __name__ == "__main__":
    t0 = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--logfiles", type=pathlib.Path, nargs="+", required=True)
    parser.add_argument("--curves", type=str, nargs="+", required=True)
    parser.add_argument("-save", action="store_true")
    args = parser.parse_args()
    main(args)
    tf = time.time()
    print(f"[done] elapsed time: {tf - t0:.2f} seconds")
