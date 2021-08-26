from reward_shaping.core.helper_fns import NormalizedReward, ThresholdIndicator


def get_normalized_reward(fun, min_r=None, max_r=None, min_r_state=None, max_r_state=None, info=None,
                          threshold=0.0, include_zero=True):
    assert min_r is None or min_r_state is None, 'if min/max_r defined, then min/max_r_state must NOT be defined'
    assert min_r_state is None or min_r is None, 'if min/max_r_state defined, then min/max_r must NOT be defined'
    assert min_r_state is None or info is not None, 'if min/max_r_state is given, info must be given to eval fun'
    # compute normalization bounds
    if min_r_state is not None and max_r_state is not None:
        min_r = fun(min_r_state, info=info, next_state=min_r_state)
        max_r = fun(max_r_state, info=info, next_state=max_r_state)
    elif min_r is not None and max_r is not None:
        pass
    else:
        raise AttributeError("either min_r and max_r defined, or min_state_r and max_state_r defined")
    # normalize reward and def indicator
    norm_fun = NormalizedReward(fun, min_r, max_r)
    indicator_fun = ThresholdIndicator(fun, threshold=threshold, include_zero=include_zero)
    return norm_fun, indicator_fun