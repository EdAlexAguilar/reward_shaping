import argparse
import pathlib

import matplotlib.pyplot as plt
import time

import numpy as np

limits = {
        "horizontal_speed":0.30,
        "vertical_speed": 0.1,
        "hull_angle": 0.08726,
        "hull_angle_speed": 0.25,
}


def _convert_dict_to_array(state):
    vars = "hull_angle,hull_angle_speed,horizontal_speed,vertical_speed," \
           "joint0_angle,joint0_angle_speed,joint1_angle,joint1_angle_speed," \
           "ground_contact_leg0,joint2_angle,joint2_angle_speed,joint3_angle," \
           "joint3_angle_speed,ground_contact_leg1,collision"
    return np.concatenate([[state[k] for k in vars.split(",")], state['lidar']])


def _convert_array_to_dict(state):
    vars = "collision,ground_contact_leg0,ground_contact_leg1,horizontal_speed,hull_angle,hull_angle_speed," \
           "joint0_angle,joint0_angle_speed,joint1_angle,joint1_angle_speed,joint2_angle,joint2_angle_speed," \
           "joint3_angle,joint3_angle_speed,lidar,vertical_speed"
    dictionary = {v: s for v, s in zip(vars.split(","), state)}
    dictionary["lidar"] = state[len(vars.split(",")):]
    return dictionary


def main(args):
    for f in args.logfiles:
        data = np.load(str(f))
        observations = data["observations"]
        ang = [_convert_array_to_dict(obs)["hull_angle"] for obs in observations]
        vx = [_convert_array_to_dict(obs)["horizontal_speed"] for obs in observations]
        vy = [_convert_array_to_dict(obs)["vertical_speed"] for obs in observations]
        angv = [_convert_array_to_dict(obs)["hull_angle_speed"] for obs in observations]
        # plot
        plt.subplot(2, 2, 1)
        plt.plot(vx, label="vx")
        plt.hlines(limits["horizontal_speed"], xmin=0, xmax=1000, label="lower - vx")
        plt.subplot(2, 2, 2)
        plt.plot(vy, label="vy")
        plt.hlines(-limits["vertical_speed"], xmin=0, xmax=1000, label="lower - vy")
        plt.hlines(limits["vertical_speed"], xmin=0, xmax=1000, label="upper - vy")
        plt.subplot(2, 2, 3)
        plt.plot(ang, label="ang")
        plt.hlines(-limits["hull_angle"], xmin=0, xmax=1000, label="lower - angle")
        plt.hlines(limits["hull_angle"], xmin=0, xmax=1000, label="upper - angle")
        plt.subplot(2, 2, 4)
        plt.plot(angv, label="angv")
        plt.hlines(-limits["hull_angle_speed"], xmin=0, xmax=1000, label="lower - angle vel")
        plt.hlines(limits["hull_angle_speed"], xmin=0, xmax=1000, label="upper - angle vel")
        plt.show()


if __name__ == "__main__":
    t0 = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--logfiles", type=pathlib.Path, nargs="+", required=True)
    parser.add_argument("-save", action="store_true")
    args = parser.parse_args()
    main(args)
    tf = time.time()
    print(f"[done] elapsed time: {tf - t0:.2f} seconds")
