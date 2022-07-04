"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

import math

import gym
import numpy as np
import pyglet
from gym import spaces
from gym.spaces import Box
from gym.utils import seeding


class DrawText:
    def __init__(self, label):
        self.label = label

    def render(self):
        self.label.draw()


class Obstacle:
    def __init__(self, axle_y, polelen, left_x, left_y, width, height):
        # cart info
        self.axle_y = axle_y
        self.polelen = polelen
        # obstacle info
        self.left_x = left_x
        self.right_x = left_x + width
        self.bottom_y = left_y
        self.top_y = left_y + height
        self.width = width
        self.height = height

    def intersect(self, x, theta):
        pole_x = x + np.sin(theta) * self.polelen
        pole_y = self.axle_y + np.cos(theta) * self.polelen
        intersect = self.left_x <= pole_x <= self.right_x and self.bottom_y <= pole_y <= self.top_y
        return intersect

    def on_left_side(self, x):
        return self.right_x < x

    def get_pole_dist(self, x, theta):
        center_x = self.left_x + (self.right_x - self.left_x) / 2.0
        center_y = self.bottom_y + (self.top_y - self.bottom_y) / 2.0
        pole_x = x + np.sin(theta) * self.polelen
        pole_y = self.axle_y + np.cos(theta) * self.polelen
        return np.linalg.norm([center_x - pole_x, center_y - pole_y])


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
        Cartpole Runs state = out of battery (battery=0)
    """
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self, task, x_limit=2.5, theta_limit=90, max_steps=200, x_target=0.0, x_target_tol=0.0,
                 theta_target=0.0, theta_target_tol=24.0, dist_target_tol=0.1,
                 cart_min_initial_offset=1.2, cart_max_initial_offset=2.0,
                 obstacle_min_w=0.5, obstacle_max_w=0.5, obstacle_min_h=0.5, obstacle_max_h=0.5, obstacle_min_dist=0.1,
                 obstacle_max_dist=0.2, feasible_height=0.97, prob_sampling_feasible=0.5, terminate_on_collision=True,
                 terminate_on_battery=False, randomize_side=True, eval=False, seed=None):
        super().__init__()
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
        self.n_resets = 0

        # Obstacle spec
        self.obstacle_min_width = obstacle_min_w
        self.obstacle_max_width = obstacle_max_w
        self.obstacle_min_height = obstacle_min_h  # this is distance to ground
        self.obstacle_max_height = obstacle_max_h  # this is distance to groun
        self.obstacle_height = 0.1  # this is the obstacle height (top_y-bottom_y)

        # Initial condition
        # cart_offset:=distance between the center of the cart and origin
        self.cart_min_offset = cart_min_initial_offset
        self.cart_max_offset = cart_max_initial_offset
        # obst_dist:=distance between the center of the cart and closest point of the obstacle
        self.obstacle_min_dist = obstacle_min_dist
        self.obstacle_max_dist = obstacle_max_dist
        self.obstacle = None
        self.is_feasible = None

        # Conditions for Episode Failure
        self.theta_threshold_radians = np.deg2rad(theta_limit)
        self.x_threshold = x_limit
        self.max_episode_steps = max_steps
        self.terminate_on_collision = terminate_on_collision
        self.terminate_on_battery = terminate_on_battery
        self.randomize_side = randomize_side
        self.step_count = 0
        self.feasible_height = feasible_height  # this is used to evaluate feasibility in overcoming obstacle TODO remove it
        self.prob_sampling_feasible = prob_sampling_feasible  # this is used to sample obstacle height TODO remove it

        # Target parameters
        self.x_target = x_target
        self.x_target_tol = x_target_tol
        self.theta_target = np.deg2rad(theta_target)
        self.theta_target_tol = np.deg2rad(theta_target_tol)
        self.dist_target_tol = dist_target_tol

        self.np_random = None
        self.viewer = None
        self.last_state = None
        self.state = None
        self.done = None
        self.steps_beyond_done = None

        self.action_space = spaces.Box(low=self.min_action, high=self.max_action, shape=(1,))
        self.seed(seed)

        self.observation_space = gym.spaces.Dict(dict(
            x=Box(low=-self.x_threshold * 2, high=self.x_threshold * 2, shape=(1,)),
            x_vel=Box(low=-np.finfo(np.float32).max, high=np.finfo(np.float32).max, shape=(1,)),
            theta=Box(low=-self.theta_threshold_radians * 2, high=self.theta_threshold_radians * 2, shape=(1,)),
            theta_vel=Box(low=-np.finfo(np.float32).max, high=np.finfo(np.float32).max, shape=(1,)),
            battery=Box(low=0, high=1, shape=(1,)),
            obstacle_left=Box(low=-self.x_threshold * 2, high=self.x_threshold * 2 - self.obstacle_max_width,
                              shape=(1,)),
            obstacle_right=Box(low=-self.x_threshold * 2 + self.obstacle_max_width, high=self.x_threshold * 2,
                               shape=(1,)),
            obstacle_bottom=Box(low=0.0, high=10.0, shape=(1,)),  # min, max are simple overapprox of the domain
            obstacle_top=Box(low=0.0, high=10.0, shape=(1,)),
            collision=Box(low=0.0, high=1.0, shape=(1,)),
        ))

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        self.action_space.seed(seed)
        return [seed]

    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg
        self.step_count += 1
        x, x_dot, theta, theta_dot, battery, obst_l, obst_r, obst_b, obst_t = tuple([
            self.state[k]
            for k
            in 'x,x_vel,theta,theta_vel,battery,obstacle_left,obstacle_right,obstacle_bottom,obstacle_top'.split(',')
        ])

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

        overcome = False if self.randomize_side else not self.obstacle.on_left_side(
            x)  # works only if starts from right
        collision = self.obstacle.intersect(x, theta)
        outside = abs(x) > self.x_threshold
        falldown = abs(theta) > self.theta_threshold_radians

        self.last_state = self.state  # used for reward shaping with potential function
        state = dict(
            x=x,
            x_vel=x_dot,
            theta=theta,
            theta_vel=theta_dot,
            battery=battery,
            obstacle_left=obst_l,
            obstacle_right=obst_r,
            obstacle_bottom=obst_b,
            obstacle_top=obst_t,
            collision=float(collision)
        )
        self.state = state

        self.done = bool(
            outside or falldown
            or self.step_count > self.max_episode_steps
            or (self.terminate_on_battery and battery <= 0)
            or (self.terminate_on_collision and collision))

        reward = self.reward()
        info = {'time': self.step_count, 'tau': self.tau,
                'max_steps': self.max_episode_steps,
                'x_limit': self.x_threshold, 'theta_limit': self.theta_threshold_radians,
                'x_target': self.x_target, 'x_target_tol': self.x_target_tol, 'dist_target_tol': self.dist_target_tol,
                'theta_target': self.theta_target, 'theta_target_tol': self.theta_target_tol,
                'pole_length': self.pole_length, 'axle_y': self.axle_y,
                'is_feasible': self.is_feasible, 'feasible_height': self.feasible_height,
                'collision': collision, 'overcome': overcome, 'outside': outside, 'falldown': falldown,
                'default_reward': reward, 'done': self.done}
        return state, reward, self.done, info

    def reward(self):
        """
        Vanilla Reward - Punish in early termination, bonus for reaching the target when it is possible
        """
        if self.done:
            if self.step_count <= self.max_episode_steps:
                return - 1.0  # early termination: either collision, outside, falldown, battery
            elif self.is_feasible and abs(self.state['x'] - self.x_target) <= self.x_target_tol:
                return + 1.0  # successfully reach the target (when possible)
        return 0.0

    def reset(self):
        """
        x_init is in [-max_offset,-min_offset] U [min_offset,max_offset]
        obstacles start on same side with a distance from the cart in [min_dist,max_dist]
        state = (x, x_dot, theta, theta_dot, battery, obstacle_left_x, obstacle_right_x, obstacle_height)
        """
        self.n_resets += 1
        self.state = {k: self.np_random.uniform(low=-0.05, high=0.05) for k in
                      self.observation_space.spaces.keys()}
        self.last_state = self.state
        self.steps_beyond_done = None
        self.done = False

        # initial position (x_init) is in [-max_offset,-min_offset] U [min_offset,max_offset]
        start = self.np_random.uniform(low=self.cart_min_offset, high=self.cart_max_offset)
        if self.randomize_side:
            if self.state['x'] > 0:
                self.state['x'] = start
            else:
                self.state['x'] = -start
        else:
            self.state['x'] = start
        # battery state
        self.state['battery'] = 1.0  # Battery Starts at 100%
        self.step_count = 0
        # sample obstacle parameters: height, width, initial distance from cart
        if self.np_random.random() <= self.prob_sampling_feasible:
            self.is_feasible = True
            obst_dist_from_ground = self.np_random.uniform(low=self.feasible_height, high=self.obstacle_max_height)
        else:
            self.is_feasible = False
            obst_dist_from_ground = self.np_random.uniform(low=self.obstacle_min_height, high=self.feasible_height)
        obstacle_width = self.np_random.uniform(low=self.obstacle_min_width, high=self.obstacle_max_width)
        # distance between cart's initial position (x_init) and the obstacle center
        distance_obst_center_to_cart = self.np_random.uniform(low=self.obstacle_min_dist + obstacle_width / 2,
                                                              high=self.obstacle_max_dist + obstacle_width / 2)
        if self.state['x'] > 0:
            left_x = self.state['x'] - distance_obst_center_to_cart - obstacle_width / 2.0
        else:
            left_x = self.state['x'] + distance_obst_center_to_cart - obstacle_width / 2.0
        axle_y = self.axle_y
        polelen = self.pole_length
        obstacle_y = axle_y + obst_dist_from_ground  # this is the bottom y of the obstacle
        self.obstacle = Obstacle(axle_y, polelen, left_x, obstacle_y, obstacle_width, self.obstacle_height)
        # store obstacle position into the state
        self.state['obstacle_left'] = self.obstacle.left_x
        self.state['obstacle_right'] = self.obstacle.right_x
        self.state['obstacle_bottom'] = self.obstacle.bottom_y
        self.state['obstacle_top'] = self.obstacle.top_y
        self.state['collision'] = float(self.obstacle.intersect(self.state['x'], self.state['theta']))
        return self.state

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400
        world_width = self.x_threshold * 2
        scale = screen_width / world_width

        carty = self.ground_y * scale
        polelen = self.pole_length * scale
        polewidth = 6.0

        cartwidth = self.cart_width * scale
        cartheight = self.cart_height * scale

        # obstacle (unscaled) dimensions
        obstacle_width = (self.obstacle.right_x - self.obstacle.left_x)

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
            # Obstacle
            l, r = -obstacle_width * scale / 2.0, obstacle_width * scale / 2.0
            t, b = -self.obstacle_height * scale / 2.0, self.obstacle_height * scale / 2.0
            obst = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.obsttrans = rendering.Transform()
            obst.set_color(*obstacle_color)
            obst.add_attr(self.obsttrans)
            self.viewer.add_geom(obst)

            # score
            text = f'episode: {self.n_resets}, time: {self.step_count}'
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

        cartx = self.state['x'] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-self.state['theta'])

        # note: translate obstacle center
        # new center:   x: left_x + obstacle_width
        #               y: cart_center_y + pole_length + obstacle_height/2
        obstx = (self.state['obstacle_left'] + obstacle_width / 2.0) * scale + screen_width / 2.0
        obsty = (self.obstacle.bottom_y + self.obstacle_height / 2.0) * scale
        self.obsttrans.set_translation(obstx, obsty)

        dist_to_ground = self.obstacle.bottom_y - self.obstacle.axle_y
        self.label.text = f'episode:{self.n_resets}, time: {self.step_count}'
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
