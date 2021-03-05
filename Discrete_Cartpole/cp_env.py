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


class CartPoleEnv(gym.Env):
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
        if goal=True
        4   Desired Goal              -2              2

    Actions:
        Type: Discrete(2)
        Num	Action
        0	Push cart to the left
        1	Push cart to the right

        Note: The amount the velocity that is reduced or increased is not
        fixed; it depends on the angle the pole is pointing. This is because
        the center of gravity of the pole increases the amount of energy needed
        to move the cart underneath it

    Reward:
        Reward is 1 for every step taken, including the termination step

    Starting State:
        All observations are assigned a uniform random value in [-0.05..0.05]

    Episode Termination:
        Pole Angle is more than limit (default 24) degrees.
        Cart Position is more than limit (default 2.4) (center of the cart reaches the edge of
        the display).
        Episode length is greater than (default 200) steps.
        Original Solved Requirements:
        Considered solved when the average reward is greater than or equal to
        195.0 over 100 consecutive trials.
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self, x_threshold=2.4, theta_threshold_deg=24, max_steps=200, goal=False):
        # Physical Constants
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = 'euler'

        # Conditions for Episode Failure
        self.theta_threshold_radians = theta_threshold_deg * math.pi / 360
        self.x_threshold = x_threshold
        self._max_episode_steps = max_steps
        self.goal = goal
        self.step_count = 0

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        high = np.array([self.x_threshold * 2,
                         np.finfo(np.float32).max,
                         self.theta_threshold_radians * 2,
                         np.finfo(np.float32).max],
                        dtype=np.float32)
        if self.goal:
            high = np.append(high, self.x_threshold*2)
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None
        self.steps_beyond_done = None
        self.state_dim = 4 # dimension of state-space

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg
        self.step_count += 1
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        temp = (force + self.polemass_length * theta_dot ** 2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == 'euler':
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        self.state = (x, x_dot, theta, theta_dot)

        done = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
            or self.step_count>self._max_episode_steps)

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
        If goal is True, then x_init is in [-1.5,-0.5] U [0.5,1.5]
        """
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(self.state_dim,))
        self.steps_beyond_done = None
        if self.goal:
            start = self.np_random.uniform(low=0.5, high=1.5)
            if self.state[0] > 0:
                self.state[0] = start
            else:
                self.state[0] = -start
        self.step_count = 0
        return np.array(self.state)

    def render(self, mode='human', end=False):
        screen_width = 600
        screen_height = 400
        world_width = self.x_threshold * 2
        scale = screen_width/world_width

        carty = 100  # TOP OF CART
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        track_interval = 0.5
        track_x_lines = [track_interval*i for i in range(int((world_width - world_width%track_interval)/track_interval+1))]
        track_x_lines = np.append(-np.flip(np.array(track_x_lines[1:])),np.array(track_x_lines))

        # COLORS
        cart_color = (0.06, 0.17, 0.26)
        bg_color = (0.75,0.85,0.95)
        track_color = (0.05, 0.05, 0.05)
        # target_color = (0.2, 0.5, 0)
        pole_color = (.5, 0, .18)
        axle_color = (0.95, .75, .15)

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
            for line_id , track_x in enumerate(track_x_lines):
                if line_id%2==0:
                    temp_draw = rendering.Line((screen_width/2 + scale*track_x, carty-20), (screen_width/2 + scale*track_x, carty+5))
                else:
                    temp_draw = rendering.Line((screen_width/2 + scale*track_x, carty-5), (screen_width/2 + scale*track_x, carty+5))
                temp_draw.set_color(*track_color)
                self.viewer.add_geom(temp_draw)
            '''
            # This was replaced - now goal is to go to center
            if self.goal:
                target_draw = rendering.make_circle(4)
                target_draw.add_attr(rendering.Transform(translation=(screen_width/2 + scale*self.target, carty-20)))
                target_draw.set_color(*target_color)
                self.viewer.add_geom(target_draw)
            '''
            # Cart
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.set_color(*cart_color)
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            self.left = rendering.make_circle(cartheight/2)
            self.lefttrans = rendering.Transform(translation=(-cartwidth*0.75+cartheight/2, 0))
            self.left.add_attr(self.carttrans)
            self.left.add_attr(self.lefttrans)
            self.left.set_color(*cart_color)
            self.viewer.add_geom(self.left)
            self.right = rendering.make_circle(cartheight/2)
            self.righttrans = rendering.Transform(translation=(cartwidth*0.75-cartheight/2, 0))
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
            self.axle = rendering.make_circle(polewidth/2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(*axle_color)
            self.viewer.add_geom(self.axle)
            self._pole_geom = pole

        if self.state is None:
            return None

        '''
        if end is True:
            self.l_eye.set_color(0.77, 0.15, .15, 0.9)
            #self.viewer.add_geom(self.l_eye)
            self.r_eye.set_color(0.77, 0.15, .15, 0.9)
        else:
            self.l_eye.set_color(0, .78, .53, 0.95)
            self.r_eye.set_color(0, .78, .53, 0.95)
        '''
        # Edit the pole polygon vertex
        pole = self._pole_geom
        l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
        pole.v = [(l, b), (l, t), (r, t), (r, b)]

        x = self.state
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
