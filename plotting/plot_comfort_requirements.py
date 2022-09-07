import argparse
import math
import pathlib
from typing import List

import matplotlib.pyplot as plt
import time

import numpy as np
from matplotlib import animation
from matplotlib.animation import FuncAnimation

LIMITS = {
    "bipedal_walker": {
        "horizontal_speed": [0.30, 1.0],
        "vertical_speed": [-0.1, 0.1],
        "hull_angle": [-0.08726, 0.08726],
        "hull_angle_speed": [-0.25, 0.25]
    },
    "racecar": {
        "steering": [-.1, +.1],
        "speed": [-1, +1],
        "velocity_x": [0.14, 0.71],
        "norm_ctrl": [0.0, 0.25]
    }
}

YLIMITS = {
    "bipedal_walker": {
        "horizontal_speed": [0.30, 1.0],
        "vertical_speed": [-0.1, 0.1],
        "hull_angle": [-0.08726, 0.08726],
        "hull_angle_speed": [-0.25, 0.25]
    },
    "racecar": {
        "speed": [-1, +1],
        "velocity_x": [-0.5, +1],
        "norm_ctrl": [-0.25, 1.5]
    }
}

labels = {
    "bipedal_walker": {
        "horizontal_speed": "Horizontal Velocity",
        "vertical_speed": "Vertical Velocity",
        "hull_angle": "Hull Angle",
        "hull_angle_speed": "Hull Angular Velocity",
    },
    "racecar": {
        "velocity_x": "Velocity",
        "steering": "Steering Angle",
        "speed": "Speed",
        "norm_ctrl": r'$\| \Delta \alpha \|$'
    },
}
show_labels = ["horizontal_speed", "hull_angle", "hull_angle_speed"]
show_labels = ["velocity_x", "norm_ctrl"]

MARGIN = 0.25  # percentage
BACKWARD_HISTORY = 1.0 #0.30
FORWARD_HISTORY = 0.10

COLORS = ['k', 'k', '#e41a1c', '#4daf4a', '#377eb8', '#984ea3', '#a65628', ]
COLORS = ['k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', ]

LINEWIDTH = 4

FIGSIZE = (10, 5)
LARGESIZE, MEDIUMSIZE, SMALLSIZE = 25, 20, 15

plt.rcParams.update({'font.size': MEDIUMSIZE})
plt.rcParams.update({'axes.titlesize': LARGESIZE})
plt.rcParams.update({'axes.labelsize': LARGESIZE})
plt.rcParams.update({'xtick.labelsize': MEDIUMSIZE})
plt.rcParams.update({'ytick.labelsize': MEDIUMSIZE})
plt.rcParams.update({'legend.fontsize': MEDIUMSIZE})
plt.rcParams.update({'figure.titlesize': MEDIUMSIZE})


def _convert_array_to_dict(state, env):
    dictionary = {}
    if env == "bipedal_walker":
        vars = "collision,ground_contact_leg0,ground_contact_leg1,horizontal_speed,hull_angle,hull_angle_speed," \
               "joint0_angle,joint0_angle_speed,joint1_angle,joint1_angle_speed,joint2_angle,joint2_angle_speed," \
               "joint3_angle,joint3_angle_speed,lidar,vertical_speed"
        dictionary = {v: s for v, s in zip(vars.split(","), state)}
        dictionary["lidar"] = state[len(vars.split(",")):]
    elif env == "racecar":
        action_vars =  "steering_2,speed_2,steering_1,speed_1,steering_0,speed_0"
        velx_vars = "velx_2,velx_1,velx_0"
        dictionary = {v: s for v, s in zip(action_vars.split(","), state)}
        dictionary.update({v: s for v, s in zip(velx_vars.split(","), state[-3:])})
        dictionary.update({
            "lidar_2": state[7:7+64],
            "lidar_1": state[7+64:7+2*64],
            "lidar_0": state[7+2*64:7+3*64],
        })
    return dictionary

def _convert_state_to_dict(state, env):
    dictionary = {}
    if env == "bipedal_walker":
        raise NotImplementedError()
    elif env == "racecar":
        dictionary = {
            "velocity_x": state["velocity_x"][0],
            "steering": state["last_actions"][-1][0],
            "speed": state["last_actions"][-1][1],
            "norm_ctrl": np.linalg.norm(state["last_actions"][-1][0] - state["last_actions"][-2][0]),
        }
    return dictionary


def parse_curve_name(filename: str):
    if "hrs_pot_nocomf" in filename or "hprs_nocomf":
        return "-Comfort"
    elif "hrs_pot" in filename or "hprs":
        return "+Comfort"
    return "Unknown"


def produce_animation(trace: np.ndarray, env: str, curve: str, var: str, color="k", save: bool = False, outfile="test.gif"):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    # prepare data
    x, yy = [], []
    values = [_convert_array_to_dict(obs, env)[var] for obs in trace]
    # animate
    n_frames = len(values)
    backward_margin = int(BACKWARD_HISTORY * n_frames)
    forward_margin = int(FORWARD_HISTORY * n_frames)

    limits = LIMITS[env]
    margin = (limits[var][1] - limits[var][0]) * MARGIN

    def animate(t):
        ax.clear()
        if t < len(values):
            x.append(t)
            yy.append(values[t])
            ax.plot(x, yy, label=curve, color=color, linewidth=LINEWIDTH)
        ax.hlines(limits[var][0], xmin=0, xmax=1000, linewidth=LINEWIDTH)
        ax.hlines(limits[var][1], xmin=0, xmax=1000, linewidth=LINEWIDTH)
        within_margin = [limits[var][0] <= y <= limits[var][1] for y in yy]
        outside_margin = [not(limits[var][0] <= y <= limits[var][1]) for y in yy]
        above_max = [y > limits[var][1] for y in yy]
        below_min = [y < limits[var][0] for y in yy]
        ax.fill_between(x, limits[var][0], limits[var][1], where=within_margin, color="#83c255", alpha=0.3)
        ax.fill_between(x, limits[var][0], limits[var][1], where=outside_margin, color="red", alpha=0.3)
        ax.fill_between(x, limits[var][0], yy, where=below_min, color="red", alpha=0.3)
        ax.fill_between(x, limits[var][1], yy, where=above_max, color="red", alpha=0.3)
        ax.set_xlim([max(0, t - backward_margin), min(t + forward_margin, n_frames)])
        ax.set_ylim([limits[var][0] - margin, limits[var][1] + margin])
        ax.set_ylim(-1, +1)
        ax.set_xlim(0, 275)
        #ax.set_xlabel("Step", horizontalalignment='right', x=1.0)
        #ax.set_ylabel(labels[var])

    anim = FuncAnimation(fig, animate, frames=n_frames, repeat=False)
    if save:
        file = pathlib.Path(outfile)
        writergif = animation.FFMpegWriter(fps=50)
        anim.save(file, writer=writergif)
    else:
        plt.show()


def produce_static_plot(trace: np.ndarray, env: str, curve: str, var: str, color="k", save: bool = False, outfile="test.pdf"):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    # prepare data

    values = [_convert_state_to_dict(info["state"], env)[var] for info in trace]
    yy = values

    limits = LIMITS[env]
    margin = (limits[var][1] - limits[var][0]) * MARGIN


    ax.clear()
    x = np.arange(0, len(values))
    ax.plot(x, yy, label=curve, color=color, linewidth=LINEWIDTH)
    ax.hlines(limits[var][0], xmin=0, xmax=1000, linewidth=LINEWIDTH)
    ax.hlines(limits[var][1], xmin=0, xmax=1000, linewidth=LINEWIDTH)

    within_margin = [limits[var][0] <= y <= limits[var][1] for y in yy]
    outside_margin = [not(limits[var][0] <= y <= limits[var][1]) for y in yy]
    above_max = [y > limits[var][1] for y in yy]
    below_min = [y < limits[var][0] for y in yy]
    ax.fill_between(x, limits[var][0], limits[var][1], where=within_margin, color="#83c255", alpha=0.4)
    ax.fill_between(x, limits[var][0], limits[var][1], where=outside_margin, color="red", alpha=0.4)
    ax.fill_between(x, limits[var][0], yy, where=below_min, color="red", alpha=0.4)
    ax.fill_between(x, limits[var][1], yy, where=above_max, color="red", alpha=0.4)
    ax.set_xlim(0, 130)
    ax.set_ylim([limits[var][0] - margin, limits[var][1] + margin])
    ax.set_ylim(YLIMITS[env][var][0], YLIMITS[env][var][1])
    ax.set_xticks([])
    ax.set_yticks([])
    #ax.set_xlabel("Step", horizontalalignment='right', x=1.0)
    ax.set_ylabel(labels[env][var])

    if save:
        file = pathlib.Path(outfile)
        plt.savefig(file)
    else:
        plt.show()


def main(args):
    traces = []  # data
    curves = args.curves  # name of the curve
    env = args.env  # name of the env
    for f in args.logfiles:
        data = np.load(str(f), allow_pickle=True)
        #traces.append(data["observations"])
        traces.append(data["infos"])
    assert len(curves) == len(args.logfiles), "nr logfile != nr curve names"

    for i, (trace, curve) in enumerate(zip(traces, curves)):
        t0 = time.time()
        print(f"[{curve}] starting")
        for j, var in enumerate(show_labels):
            print(f"\t{var}")
            color = COLORS[i]
            outfile = f"comfort_plot_{curve}_{labels[env][var]}_{int(time.time())}"
            if args.static:
                produce_static_plot(trace, env, curve, var, color=color, save=args.save, outfile=f"{outfile}.pdf")
            else:
                produce_animation(trace, env, curve, var, color=color, save=args.save, outfile=f"{outfile}.gif")
        tf = time.time()
        print(f"[{curve}] done in: {tf - t0:.2f} seconds")


if __name__ == "__main__":
    t0 = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--logfiles", type=pathlib.Path, nargs="+", required=True)
    parser.add_argument("--curves", type=str, nargs="+", required=True)
    parser.add_argument("--env", type=str, default="bipedal_walker")
    parser.add_argument("-save", action="store_true")
    parser.add_argument("-static", action="store_true", help="disable animation, static plot of entire trace")
    args = parser.parse_args()
    main(args)
    tf = time.time()
    print(f"[done] elapsed time: {tf - t0:.2f} seconds")
