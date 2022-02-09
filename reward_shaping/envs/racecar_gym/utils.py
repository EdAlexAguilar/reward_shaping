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
