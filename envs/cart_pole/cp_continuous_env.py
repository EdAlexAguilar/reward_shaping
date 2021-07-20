"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

import math
import gym
import pyglet
from gym import spaces, logger
from gym.utils import seeding
import numpy as np


class DrawText:
    def __init__(self, label: pyglet.text.Label):
        self.label = label

    def render(self):
        self.label.draw()


class CartPoleContEnv(gym.Env):
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
    Actions:
        Type: Box   (Continuous)
        Low = -1
        High = 1


    Episode Termination:
        Pole Angle is more than limit (default 90) degrees.
        Cart Position is more than limit (default 2.5) (center of the cart reaches the edge of
        the display).
        Episode length is greater than (default 200) steps.
        Cartpole Runs state = out of battery (battery=0) if enabled
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self, task, x_limit=2.5, theta_limit=90, max_steps=200,
                 x_target=0.0, x_target_tol=0.0, theta_target=0.0, theta_target_tol=24.0,
                 cart_min_initial_offset=1.2, cart_max_initial_offset=2.0,
                 terminate_on_battery=False, randomize_side=True,
                 eval=False, seed=None):
        # Physical Constants
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.ground_y = 1.0
        self.cart_width = 0.40
        self.cart_height = 0.25
        self.length = 0.5  # actually half the pole's length
        self.pole_length = 2 * self.length
        self.axle_y = self.ground_y + self.cart_height / 4.0
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 30.0  # 10.0
        self.tau = 0.02  # seconds between state updates
        self.battery_consumption = 0.075
        self.min_action = -1.0
        self.max_action = 1.0

        self.task = task
        self.eval_env = eval
        self.n_reset = 0

        # Initial condition
        # cart_offset:=distance between the center of the cart and origin
        self.cart_min_offset = cart_min_initial_offset
        self.cart_max_offset = cart_max_initial_offset

        # Conditions for Episode Failure
        self.theta_threshold_radians = np.deg2rad(theta_limit)
        self.x_threshold = x_limit
        self.max_episode_steps = max_steps
        self.terminate_on_battery = terminate_on_battery
        self.randomize_side = randomize_side
        self.step_count = 0

        # Target parameters
        self.x_target = x_target
        self.x_target_tol = x_target_tol
        self.theta_target = np.deg2rad(theta_target)
        self.theta_target_tol = np.deg2rad(theta_target_tol)

        self.seed(seed)
        self.viewer = None
        self.last_state = None
        self.state = None
        self.steps_beyond_done = None
        self.state_dim = self.observation_space.shape[0]  # dimension of state-space

        # for rendering
        self.rew = 0.0
        self.ret = 0.0
        self.safety_tot = 0.0
        self.target_tot = 0.0
        self.comfort_tot = 0.0

        # for evaluation
        self.episode = {v: [] for v in self.monitoring_variables}
        self.last_complete_episode = None

    @property
    def observation_space(self):
        low = np.array([-self.x_threshold * 2,
                        -np.finfo(np.float32).max,
                        -self.theta_threshold_radians * 2,
                        -np.finfo(np.float32).max,
                        0],
                       dtype=np.float32)
        high = np.array([self.x_threshold * 2,
                         np.finfo(np.float32).max,
                         self.theta_threshold_radians * 2,
                         np.finfo(np.float32).max,
                         1],
                        dtype=np.float32)
        return spaces.Box(low, high, dtype=np.float32)

    @property
    def action_space(self):
        return spaces.Box(low=self.min_action, high=self.max_action, shape=(1,))

    @property
    def monitoring_variables(self):
        return ['time', 'dist_target_x', 'dist_target_theta', 'x', 'theta', 'pole_x', 'pole_y']

    @property
    def monitoring_types(self):
        return ['int', 'float', 'float', 'float', 'float', 'float', 'float', 'float']

    @property
    def monitoring_spec(self):
        # safety specs
        no_falldown = f"always(abs(theta) <= {self.theta_threshold_radians})"
        no_outside = f"always(abs(x) <= {self.x_threshold})"
        # target spec
        target_requirements = f"eventually(always(dist_target_x <= {self.x_target_tol}))"
        balance_requirements = f"always(dist_target_theta <= {self.theta_target_tol})"
        # all together
        if self.task == 'balance':
            safety_requirements = f"({no_falldown}) and ({no_outside})"
            spec = f"({safety_requirements}) and ({balance_requirements})"
        elif self.task == 'target':
            safety_requirements = f"({no_falldown}) and ({no_outside})"
            spec = f"({safety_requirements}) and ({balance_requirements}) and ({target_requirements})"
        else:
            raise NotImplemented(f'no formal spec for task {self.task}')
        return spec

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg
        self.step_count += 1
        x, x_dot, theta, theta_dot, battery = self.state

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

        self.last_state = self.state
        self.state = (x, x_dot, theta, theta_dot, battery)

        self.done = bool(
            abs(x) > self.x_threshold
            or abs(theta) > self.theta_threshold_radians
            or self.step_count > self.max_episode_steps
            or (self.terminate_on_battery and battery <= 0))

        self.rew = self.reward()

        self.safety_tot = 0.0
        self.target_tot = 0.0
        self.comfort_tot = 0.0

        self._update_episode()  # update episode for monitoring
        return np.array(self.state), self.rew, self.done, {}

    def _update_episode(self):
        # compute monitoring variables
        x, theta = self.state[0], self.state[2]
        dist_target_x = abs(x - self.x_target)
        dist_target_theta = abs(theta - self.theta_target)
        # extend episode history
        self.episode['time'].append(self.step_count)
        self.episode['dist_target_x'].append(dist_target_x)
        self.episode['dist_target_theta'].append(dist_target_theta)
        self.episode['x'].append(x)
        self.episode['theta'].append(theta)
        self.episode['pole_x'].append(x + self.pole_length * np.sin(theta))
        self.episode['pole_y'].append(self.axle_y + self.pole_length * np.cos(theta))
        # eventually store if done
        if self.done:
            self.last_complete_episode = self.episode

    def compute_episode_robustness(self, episode, spec):
        # compute robustness
        import rtamt
        spec = rtamt.STLSpecification()
        for v, t in zip(self.monitoring_variables, self.monitoring_types):
            spec.declare_var(v, f'{t}')
        spec.spec = self.monitoring_spec
        spec.parse()
        # preprocess format, evaluate, post process
        robustness_trace = spec.evaluate(episode)
        return robustness_trace[0][1]

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
        """
        self.n_reset += 1
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(self.state_dim,)).astype(np.float32)

        self.steps_beyond_done = None
        self.done = False
        # reset rewards
        self.rew = 0.0
        self.ret = 0.0

        self.safety_tot = 0.0
        self.target_tot = 0.0
        self.comfort_tot = 0.0

        if self.eval_env:
            # initial position (x_init) is fixed
            self.state[0] = (-1) ** self.n_reset * abs(self.state[0])  # if eval, then side is alternated
            start = self.cart_min_offset + (self.cart_max_offset - self.cart_min_offset) / 2.0
        else:
            # initial position (x_init) is in [-max_offset,-min_offset] U [min_offset,max_offset]
            start = self.np_random.uniform(low=self.cart_min_offset, high=self.cart_max_offset)

        if self.randomize_side:
            if self.state[0] > 0:
                self.state[0] = start
            else:
                self.state[0] = -start
        else:
            self.state[0] = start
        # battery state
        self.state[4] = 1  # Battery Starts at 100%
        self.step_count = 0
        self.last_state = self.state
        # reset episode
        self.episode = {v: [] for v in self.monitoring_variables}
        return np.array(self.state)

    def render(self, mode='human', end=False):
        screen_width = 600
        screen_height = 400
        world_width = self.x_threshold * 2
        scale = screen_width / world_width

        carty = self.ground_y * scale
        polelen = self.pole_length * scale
        polewidth = 6.0

        cartwidth = self.cart_width * scale
        cartheight = self.cart_height * scale

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

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            # Track and background must be drawn fist
            BG_l, BG_r, BG_t, BG_b = 0, screen_width, 0, screen_height
            BG = rendering.FilledPolygon([(BG_l, BG_b), (BG_l, BG_t), (BG_r, BG_t), (BG_r, BG_b)])
            BG.set_color(*bg_color)
            self.viewer.add_geom(BG)
            self.track1 = rendering.Line((0, scale * self.ground_y), (screen_width, scale * self.ground_y))
            self.track1.set_color(*track_color)
            self.viewer.add_geom(self.track1)

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
            l, r, t, b = -polewidth / 2, polewidth / 2, polelen, 0.0
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

            # score
            text = f'safety = {self.safety_tot:.2f}, target = {self.target_tot:.2f}, comfort: {self.comfort_tot:.2f}\n' \
                   f'time: {self.step_count}, reward = {self.rew:.2f}, return = {self.ret:.2f}'
            self.label = pyglet.text.Label(text, font_size=15, multiline=True, width=1000,
                                           x=5, y=20, anchor_x='left', anchor_y='bottom',
                                           color=(255, 255, 255, 255))
            self.label.draw()
            self.viewer.add_geom(DrawText(self.label))

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

        self.label.text = f'safety = {self.safety_tot:.2f}, target = {self.target_tot:.2f}, comfort: {self.comfort_tot:.2f}\n' \
                          f'time: {self.step_count}, reward = {self.rew:.2f}, return = {self.ret:.2f}'
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
