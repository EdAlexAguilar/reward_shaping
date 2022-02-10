from numba import njit
import numpy as np


@njit(fastmath=False, cache=True)
def polar2cartesian(dist, angle, n_bins, res):
    occupancy_map = np.zeros(shape=(n_bins, n_bins), dtype=np.uint8)
    xx = dist * np.cos(angle)
    yy = dist * np.sin(angle)
    xi, yi = np.floor(xx / res), np.floor(yy / res)
    for px, py in zip(xi, yi):
        row = min(max(n_bins // 2 + py, 0), n_bins - 1)
        col = min(max(n_bins // 2 + px, 0), n_bins - 1)
        if row < n_bins - 1 and col < n_bins - 1:
            # in this way, then >max_range we dont fill the occupancy map in order to let a visible gap
            occupancy_map[int(row), int(col)] = 255
    return np.expand_dims(occupancy_map, 0)


@njit(fastmath=False, cache=True)
def dist_to_wall(scan, max_halfwidth):
    """ compute distance to the right wall as in follow the wall, return normalized value according to `max_halfwidt`
        resource: https://f1tenth-coursekit.readthedocs.io/en/latest/assignments/labs/lab3.html#doc-lab3 """
    # find dist a, b for angles 45, 90 deg from x-axis respectively
    delta_angle = np.deg2rad(270) / len(scan)  # difference between consecutive angles (radians)
    a_ray = int(len(scan) / 2 + (np.deg2rad(45) / delta_angle))  # len(scan)/2 is the frontal ray (x-axis)
    b_ray = int(len(scan) / 2 + (np.deg2rad(90) / delta_angle))  # then move from it towards left of 45, 90 deg
    a, b = scan[a_ray], scan[b_ray]
    theta = (b_ray - a_ray) * delta_angle  # angle between a, b segments
    # compute angle alpha between b and normal to right wall (Eq. 3.1)
    alpha = np.arctan((a * np.sin(theta) - b) / (a * np.sin(theta)))
    # compute distance
    d = b * np.cos(alpha)
    d = 2 * max_halfwidth if d > 2 * max_halfwidth else d
    return d
