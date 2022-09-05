from typing import Union

import numpy as np


def clip_and_norm(v: Union[int, float], minv: Union[int, float], maxv: Union[int, float]):
    """
    utility function which returns the normalized value v' in [0, 1].

    @params: value `v` before normalization,
    @params: `minv`, `maxv` extreme values of the domain.
    """
    if np.abs(minv - maxv) <= 0.000001:
        return 0.0
    return (np.clip(v, minv, maxv) - minv) / (maxv - minv)
