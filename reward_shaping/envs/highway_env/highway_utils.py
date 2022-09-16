from reward_shaping.envs.highway_env import highway_constants as c


def safe_long_dist(v1, v2):

    if same_lane(v1, v2) and behind(v1, v2):
        tau = c.REACTION_TIME
        acc_max = c.ACCEL_MAX
        brake_min = c.BRAKE_MIN
        brake_max = c.BRAKE_MAX
        d_safe = v1[3] * tau + 0.5 * acc_max * tau ** 2 + (v2[3] + tau * acc_max) ** 2 / (      # vi[3] ~ v_x ~ v_lon
                2 * brake_min) - v1[3] ** 2 / (2 * brake_max)
        return c.LON_DIST_NORMALIZER*d_safe
    else:
        return -float('inf')


def safe_lat_dist(v1, v2):
    if same_lane(v1, v2):
        return c.LANE_WIDTH
    if left_lane(v1, v2):
        left_car = v1
        right_car = v2
    else:
        left_car = v2
        right_car = v1

    tau = c.REACTION_TIME
    lat_dist_margin = c.LAT_DIST_MARGIN     # mu
    lat_acc_max = c.LAT_ACCEL_MAX
    lat_brake_min = c.LAT_MIN_BR
    right_braking_dist = right_car[4] * tau + 0.5 * lat_acc_max * tau ** 2 +\
                         (right_car[3] * tau * lat_acc_max) **2 / (2 * lat_brake_min)   # car[3] ~ v_x ~ v_lon
    left_braking_dist = left_car[4] * tau - 0.5 * lat_acc_max * tau ** 2 -\
                         (left_car[3] * tau * lat_acc_max) **2 / (2 * lat_brake_min)    # car[4] ~ v_y ~ v_lat
    d_safe = lat_dist_margin + right_braking_dist - left_braking_dist
    return c.LAT_DIST_NORMALIZER*d_safe


def longitudinal_distance(v1, v2):
    """
    The longitudinal distance between two cars v1 and v2
    v1 is intended to be BEHIND v2
    """
    # First: Check if v2 is geometrically in front of v1 or not
    if not behind(v1, v2):
        return float('inf')  # an arbitrary large value.
    else:
        dist = longitudinal_road_distance(v1, v2)
        dist -= c.VEHICLE_LENGTH  # (ego_length + actor_length) approximated
        return max(dist, 0)


def lateral_distance(v1, v2):
    if same_lane(v1, v2):
        return 0.0
    if left_lane(v1, v2):
        left_car = v1
        right_car = v2
    else:
        left_car = v2
        right_car = v1
    dist = lateral_road_distance(left_car, right_car)
    dist -= c.VEHICLE_WIDTH # (ego_width + actor_width) approximated
    return max(dist, 0)


def legal_speed_limit(v1, v2):
    '''
    Currently This is a pairwise contract (since need to know if road is clear)
    if ( d_lon > d_safe AND behind(v1, v2) ) then |v2.vel - v_speedlimit| < tolerance
    v_speedlimit == monitor.constants.SPEED_LIMIT
    tolerance == monitor.constants.SPEED_LIMIT_TOL
    '''

    v2_in_front = behind(v1, v2) and same_lane(v1, v2)
    dist = longitudinal_distance(v1, v2)
    safe_dist = safe_long_dist(v1, v2)
    vel_diff = abs(v1[3] - c.HARD_SPEED_LIMIT)
    return v2_in_front, dist, safe_dist, vel_diff


def legal_overtake(v1, v2):
    '''
        Currently This is a pairwise contract
    if  V2_ON_LEFT then vel_diff > 0
    v2_on_left : {True = + infty ; False = - Infty}
    vel_diff = v2.vel - v1.vel
    '''
    v2_on_left = left_lane(v2, v1) and in_vicinity(v1, v2)
    vel_diff = v2[3] - v1[3]
    return v2_on_left, vel_diff


def lateral_road_distance(v1, v2):
    # TODO: Ask Edgar & Dejan : Doesn't need abs?
    return v2[2] - v1[2]    # v[2] ~ y


def longitudinal_road_distance(v1, v2):
    return v2[1] - v1[1]    # v[1] ~ x


def behind(v1, v2):
    return True if v2[1] > v1[1] else False     # v[1] ~ x


def same_lane(v1, v2):
    return True if lane_id(v1[2]) == lane_id(v2[2]) else False      # v[2] ~ y


def left_lane(v1, v2):
    return True if lane_id(v1[2]) < lane_id(v2[2]) else False


def lane_id(y):
    return int(y/c.LANE_WIDTH)


def in_vicinity(v1, v2, vicinity=60):
    if (longitudinal_distance(v1, v2) < vicinity) or\
      (longitudinal_distance(v2, v1) < vicinity):
        return True
    else:
        return False