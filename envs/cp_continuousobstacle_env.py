"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R

Modification of OpenAI CartPole environment
-EA Aguilar

"""

import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np


class Obstacle():
    def __init__(self, left_x, width, height):
        self.left_x = left_x
        self.right_x = left_x + width
        self.height = height

    def intersect(self, x, theta):
        """
        x, theta cartpole coords
        + dist if pole is outside obstacle
        - dist if pole is inside obstacle
        # ASSUMES pole self.length = 0.5  # actually half the pole's length
        """
        if theta == 0:
            theta = 1e-5
        y_left = (1 / np.tan(theta)) * (self.left_x - x) - (1 - self.height)
        y_right = (1 / np.tan(theta)) * (self.right_x - x) - (1 - self.height)
        return np.sign(y_left * y_right) * np.min([abs(y_left), abs(y_right)]) < 0.0


class CartPoleContObsEnv(gym.Env):
    """
    Description:
        A pole is attached by an un-actuated joint to a cart, which moves along
        a frictionless track. The pendulum starts upright, and the goal is to
        prevent it from falling over by increasing and reducing the cart's
        velocity.

    Source:
        This environment corresponds to the version of the cart-pole problem
        described by Barto, Sutton, and Anderson

    Observation:
        Type: Box(4)
        Num	Observation               Min             Max
        0	Cart Position             -4.8            4.8
        1	Cart Velocity             -Inf            Inf
        2	Pole Angle                -24 deg         24 deg
        3	Pole Velocity At Tip      -Inf            Inf
        4   Battery Level Percent       0              1
        5   Obstacle Left
        6   Obstacle Right
        7   Obstacle Height from top
    Actions:
        Type: Box   (Continuous)
        Low = -1
        High = 1


    Episode Termination:
        Pole Angle is more than limit (default 24) degrees.
        Cart Position is more than limit (default 2.4) (center of the cart reaches the edge of
        the display).
        Episode length is greater than (default 200) steps.
        Cartpole Runs out of battery (battery=0)
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self, x_threshold=2.5, theta_threshold_deg=24, max_steps=200,
                 cart_min_initial_offset=1.2, cart_max_initial_offset=2.0,
                 obstacle_min_w=0.5, obstacle_max_w=0.5, obstacle_min_h=0.5, obstacle_max_h=0.5,
                 obstacle_min_dist=0.1, obstacle_max_dist=0.2):
        # Physical Constants
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 30.0  # 10.0
        self.tau = 0.02  # seconds between state updates
        self.battery_consumption = 0.075
        self.obstacle_min_width = obstacle_min_w
        self.obstacle_max_width = obstacle_max_w
        self.obstacle_min_height = obstacle_min_h
        self.obstacle_max_height = obstacle_max_h
        self.min_action = -1.0
        self.max_action = 1.0

        # Initial condition
        # cart_offset:=distance between the center of the cart and origin
        self.cart_min_offset = cart_min_initial_offset
        self.cart_max_offset = cart_max_initial_offset
        # obst_dist:=distance between the center of the cart and closest point of the obstacle
        self.obstacle_min_dist = obstacle_min_dist
        self.obstacle_max_dist = obstacle_max_dist

        # Conditions for Episode Failure
        self.theta_threshold_radians = theta_threshold_deg * math.pi / 360
        self.x_threshold = x_threshold
        self._max_episode_steps = max_steps
        self.step_count = 0

        high = np.array([self.x_threshold * 2,
                         np.finfo(np.float32).max,
                         self.theta_threshold_radians * 2,
                         np.finfo(np.float32).max,
                         1,
                         self.x_threshold * 2 - self.obstacle_max_width,
                         self.x_threshold * 2,
                         self.obstacle_max_height],
                        dtype=np.float32)
        low = np.array([-self.x_threshold * 2,
                        -np.finfo(np.float32).max,
                        -self.theta_threshold_radians * 2,
                        -np.finfo(np.float32).max,
                        0,
                        -self.x_threshold * 2,
                        -self.x_threshold * 2 + self.obstacle_max_width,
                        self.obstacle_min_height],
                       dtype=np.float32)

        self.action_space = spaces.Box(
            low=self.min_action,
            high=self.max_action,
            shape=(1,))

        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None
        self.steps_beyond_done = None
        self.state_dim = len(high)  # dimension of state-space

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg
        self.step_count += 1
        x, x_dot, theta, theta_dot, battery, obst_l, obst_r, obst_h = self.state

        force = float(action * self.force_mag)
        battery -= float(abs(action) * self.battery_consumption)

        # Physics Step
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = (force + self.polemass_length * theta_dot ** 2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
                self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc

        self.state = (x, x_dot, theta, theta_dot, battery, obst_l, obst_r, obst_h)

        done = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
            or self.step_count > self._max_episode_steps
            or battery <= 0
            or self.obstacle.intersect(x, theta))

        return np.array(self.state), self.reward(), done, {}

    def reward(self):
        """
        Vanilla Reward - To be overridden
        """
        if abs(self.state[2]) < self.theta_threshold_radians:
            return 1.0
        else:
            return 0.0

    def reset(self):
        """
        x_init is in [-max_offset,-min_offset] U [min_offset,max_offset]
        obstacles start on same side with a distance from the cart in [min_dist,max_dist]
        state = (x, x_dot, theta, theta_dot, battery, obstacle_left_x, obstacle_right_x, obstacle_height)
        """
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(self.state_dim,))
        self.steps_beyond_done = None
        # initial position (x_init) is in [-max_offset,-min_offset] U [min_offset,max_offset]
        start = self.np_random.uniform(low=self.cart_min_offset, high=self.cart_max_offset)
        if self.state[0] > 0:
            self.state[0] = start
        else:
            self.state[0] = -start
        # battery state
        self.state[4] = 1  # Battery Starts at 100%
        self.step_count = 0
        # sample obstacle parameters: height, width, initial distance from cart
        obstacle_height = self.np_random.uniform(low=self.obstacle_min_height, high=self.obstacle_max_height)
        obstacle_width = self.np_random.uniform(low=self.obstacle_min_width, high=self.obstacle_max_width)
        # distance between cart's initial position (x_init) and the obstacle center
        distance_obst_center_to_cart = self.np_random.uniform(low=self.obstacle_min_dist + obstacle_width / 2,
                                                              high=self.obstacle_max_dist + obstacle_width / 2)
        if self.state[0] > 0:
            left_x = self.state[0] - distance_obst_center_to_cart - obstacle_width / 2.0
        else:
            left_x = self.state[0] + distance_obst_center_to_cart - obstacle_width / 2.0
        self.obstacle = Obstacle(left_x, obstacle_width, obstacle_height)
        # store obstacle position into the state
        self.state[5] = self.obstacle.left_x
        self.state[6] = self.obstacle.right_x
        self.state[7] = obstacle_height
        return np.array(self.state)

    def set_obstacle_width_height(self, width, height):
        self.obstacle_max_width = width
        self.obstacle_max_height = height

    def render(self, mode='human', end=False):
        screen_width = 600
        screen_height = 400
        world_width = self.x_threshold * 2
        scale = screen_width / world_width

        carty = 100  # TOP OF CART
        polewidth = 6.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        # obstacle (unscaled) dimensions
        obstacle_width = (self.obstacle.right_x - self.obstacle.left_x)
        obstacle_height = self.obstacle.height

        track_interval = 0.5
        track_x_lines = [track_interval * i for i in
                         range(int((world_width - world_width % track_interval) / track_interval + 1))]
        track_x_lines = np.append(-np.flip(np.array(track_x_lines[1:])), np.array(track_x_lines))

        # COLORS
        cart_color = (0.06, 0.17, 0.26)
        bg_color = (0.7, 0.80, 0.90)
        track_color = (0.05, 0.05, 0.05)
        # target_color = (0.2, 0.5, 0)
        pole_color = (.5, 0, .18)
        axle_color = (0.9, .7, .1)
        obstacle_color = (0.05, 0.35, 0.1)

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            # Track and background must be drawn fist
            BG_l, BG_r, BG_t, BG_b = 0, screen_width, 0, screen_height
            BG = rendering.FilledPolygon([(BG_l, BG_b), (BG_l, BG_t), (BG_r, BG_t), (BG_r, BG_b)])
            BG.set_color(*bg_color)
            self.viewer.add_geom(BG)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(*track_color)
            self.viewer.add_geom(self.track)
            for line_id, track_x in enumerate(track_x_lines):
                if line_id % 2 == 0:
                    temp_draw = rendering.Line((screen_width / 2 + scale * track_x, carty - 20),
                                               (screen_width / 2 + scale * track_x, carty + 5))
                else:
                    temp_draw = rendering.Line((screen_width / 2 + scale * track_x, carty - 5),
                                               (screen_width / 2 + scale * track_x, carty + 5))
                temp_draw.set_color(*track_color)
                self.viewer.add_geom(temp_draw)

            # Cart
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.set_color(*cart_color)
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            self.left = rendering.make_circle(cartheight / 2)
            self.lefttrans = rendering.Transform(translation=(-cartwidth * 0.75 + cartheight / 2, 0))
            self.left.add_attr(self.carttrans)
            self.left.add_attr(self.lefttrans)
            self.left.set_color(*cart_color)
            self.viewer.add_geom(self.left)
            self.right = rendering.make_circle(cartheight / 2)
            self.righttrans = rendering.Transform(translation=(cartwidth * 0.75 - cartheight / 2, 0))
            self.right.add_attr(self.carttrans)
            self.right.add_attr(self.righttrans)
            self.right.set_color(*cart_color)
            self.viewer.add_geom(self.right)
            # Pole
            l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(*pole_color)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            # Axle
            self.axle = rendering.make_circle(polewidth / 2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(*axle_color)
            self.viewer.add_geom(self.axle)
            self._pole_geom = pole
            # Obstacle
            l, r = -obstacle_width * scale / 2.0, obstacle_width * scale / 2.0
            t, b = -obstacle_height * scale / 2.0, obstacle_height * scale / 2.0
            obst = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.obsttrans = rendering.Transform()
            obst.set_color(*obstacle_color)
            obst.add_attr(self.obsttrans)
            self.viewer.add_geom(obst)

        if self.state is None:
            return None

        # Edit the pole polygon vertex
        pole = self._pole_geom
        l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
        pole.v = [(l, b), (l, t), (r, t), (r, b)]

        x = self.state
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        # note: translate obstacle center
        # new center:   x: left_x + obstacle_width
        #               y: cart_center_y + pole_length + obstacle_height/2
        obstx = (self.state[5] + obstacle_width / 2.0) * scale + screen_width / 2.0
        obsty = carty + polelen + (obstacle_height / 2.0 * scale)
        self.obsttrans.set_translation(obstx, obsty)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
