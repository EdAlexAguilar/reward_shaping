"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R

Modification of OpenAI CartPole environment
-EA Aguilar

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


class Obstacle():
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

    def __init__(self, task, x_limit=2.5, theta_limit=90, max_steps=200,
                 x_target=0.0, x_target_tol=0.0, theta_target=0.0, theta_target_tol=24.0,
                 cart_min_initial_offset=1.2, cart_max_initial_offset=2.0,
                 obstacle_min_w=0.5, obstacle_max_w=0.5, obstacle_min_h=0.5, obstacle_max_h=0.5,
                 obstacle_min_dist=0.1, obstacle_max_dist=0.2, feasible_height=0.97,
                 terminate_on_collision=True, terminate_on_battery=False, randomize_side=True,
                 eval=False, seed=None):
        self.task = task
        self.eval = eval
        self.n_resets = 0
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

        # Obstacle spec
        self.obstacle_min_width = obstacle_min_w
        self.obstacle_max_width = obstacle_max_w
        self.obstacle_min_height = obstacle_min_h
        self.obstacle_max_height = obstacle_max_h

        # Initial condition
        # cart_offset:=distance between the center of the cart and origin
        self.cart_min_offset = cart_min_initial_offset
        self.cart_max_offset = cart_max_initial_offset
        # obst_dist:=distance between the center of the cart and closest point of the obstacle
        self.obstacle_min_dist = obstacle_min_dist
        self.obstacle_max_dist = obstacle_max_dist
        self.obstacle = None

        # Conditions for Episode Failure
        self.theta_threshold_radians = np.deg2rad(theta_limit)
        self.x_threshold = x_limit
        self.max_episode_steps = max_steps
        self.terminate_on_collision = terminate_on_collision
        self.terminate_on_battery = terminate_on_battery
        self.randomize_side = randomize_side
        self.step_count = 0
        self.feasible_height = feasible_height  # this is used to evaluate feasibility in overcoming obstacle

        # Target parameters
        self.x_target = x_target
        self.x_target_tol = x_target_tol
        self.theta_target = np.deg2rad(theta_target)
        self.theta_target_tol = np.deg2rad(theta_target_tol)

        # for rendering
        self.rew = 0.0
        self.ret = 0.0
        self.safety_tot = 0.0
        self.target_tot = 0.0
        self.comfort_tot = 0.0

        self.viewer = None
        self.last_state = None
        self.state = None
        self.initial_state = None
        self.steps_beyond_done = None
        self.state_dim = self.observation_space.shape[0]  # dimension of state-space

        # for monitoring
        self.episode = {v: [] for v in self.monitoring_variables}
        self.last_complete_episode = None

        self.action_space = spaces.Box(low=self.min_action, high=self.max_action, shape=(1,))
        self.seed(seed)

    @property
    def monitoring_variables(self):
        return ['time', 'collision', 'falldown', 'outside',
                'dist_target_x', 'dist_obstacle', 'dist_target_theta',
                'x', 'theta', 'pole_x', 'pole_y', 'obst_y_from_axle',
                'obst_left_x', 'obst_right_x', 'obst_bottom_y', 'obst_top_y']

    @property
    def monitoring_types(self):
        return ['int', 'int', 'int', 'int',
                'float', 'float', 'float', 'float', 'float', 'float',
                'float', 'float', 'float', 'float', 'float', 'float']

    @property
    def monitoring_specs(self):
        # aux spec
        obst_intersect_polex = f"(obst_left_x < pole_x) and (pole_x < obst_right_x)"
        obst_intersect_poley = f"(obst_bottom_y < pole_y) and (pole_y < obst_top_y)"
        # safety specs cont
        no_falldown = f"always(abs(theta) <= {self.theta_threshold_radians})"
        no_outside = f"always(abs(x) <= {self.x_threshold})"
        no_collision = f"always(not(({obst_intersect_polex}) and ({obst_intersect_poley})))"
        safety_requirements = f"({no_falldown}) and ({no_outside}) and ({no_collision})"
        # safety specs bool
        no_falldown_bool = f"always(falldown >= 0.0)"
        no_outside_bool = f"always(outside >= 0)"
        no_collision_bool = f"always(collision >= 0)"
        safety_requirements_bool = f"({no_falldown_bool}) and ({no_outside_bool}) and ({no_collision_bool})"
        # target spec
        target_requirement = f"eventually(always(dist_target_x <= {self.x_target_tol}))"
        balance_requirement = f"eventually(always(dist_target_theta <= {self.theta_target_tol}))"
        # all together
        if self.obstacle.bottom_y - self.axle_y >= self.feasible_height:
            spec_cont = f"({safety_requirements}) and ({target_requirement}) and ({balance_requirement})"
            spec_bool = f"({safety_requirements_bool}) and ({target_requirement}) and ({balance_requirement})"
        else:
            spec_cont = f"({safety_requirements}) and ({balance_requirement})"
            spec_bool = f"({safety_requirements_bool}) and ({balance_requirement})"
        return spec_cont, spec_bool

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        self.action_space.seed(seed)
        return [seed]

    @property
    def observation_space(self):
        low = np.array([-self.x_threshold * 2,
                        -np.finfo(np.float32).max,
                        -self.theta_threshold_radians * 2,
                        -np.finfo(np.float32).max,
                        0,
                        -self.x_threshold * 2,
                        -self.x_threshold * 2 + self.obstacle_max_width,
                        self.obstacle_min_height],
                       dtype=np.float32)
        high = np.array([self.x_threshold * 2,
                         np.finfo(np.float32).max,
                         self.theta_threshold_radians * 2,
                         np.finfo(np.float32).max,
                         1,
                         self.x_threshold * 2 - self.obstacle_max_width,
                         self.x_threshold * 2,
                         self.obstacle_max_height],
                        dtype=np.float32)
        return spaces.Box(low, high, dtype=np.float32)

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

        self.last_state = self.state
        self.state = (x, x_dot, theta, theta_dot, battery, obst_l, obst_r, obst_h)

        self.done = bool(
            abs(x) > self.x_threshold
            or abs(theta) > self.theta_threshold_radians
            or self.step_count > self.max_episode_steps
            or (self.terminate_on_battery and battery <= 0)
            or (self.terminate_on_collision and self.obstacle.intersect(x, theta)))

        self.rew = self.reward()

        # update scores
        self.x_target_score = self.x_target_tol - abs(x - self.x_target)
        self.theta_target_score = self.theta_target_tol - abs(theta - self.theta_target)
        self.collision_score = -1.0 if self.obstacle.intersect(x, theta) else 0.0
        self.falldown_score = -1.0 if abs(theta) > self.theta_threshold_radians else 0.0

        self.safety_tot = 0.0
        self.target_tot = 0.0
        self.comfort_tot = 0.0

        self._update_episode()  # update episode for monitoring
        return np.array(self.state), self.rew, self.done, {}

    def _update_episode(self):
        # compute monitoring variables
        x, theta = self.state[0], self.state[2]
        collision = -3.0 if self.obstacle.intersect(x, theta) else +3.0
        falldown = -3.0 if abs(theta) > self.theta_threshold_radians else +3.0
        outside = -3.0 if abs(x) > self.x_threshold else +3.0
        dist_target_x = abs(x - self.x_target)
        dist_target_theta = abs(theta - self.theta_target)
        dist_obstacle = abs(x - (self.obstacle.left_x + (self.obstacle.right_x - self.obstacle.left_x) / 2.0))
        # extend episode history
        self.episode['time'].append(self.step_count)
        self.episode['collision'].append(collision)
        self.episode['falldown'].append(falldown)
        self.episode['outside'].append(outside)
        self.episode['dist_target_x'].append(dist_target_x)
        self.episode['dist_target_theta'].append(dist_target_theta)
        self.episode['dist_obstacle'].append(dist_obstacle)
        self.episode['x'].append(x)
        self.episode['theta'].append(theta)
        self.episode['pole_x'].append(x + self.pole_length * np.sin(theta))
        self.episode['pole_y'].append(self.axle_y + self.pole_length * np.cos(theta))
        self.episode['obst_left_x'].append(self.obstacle.left_x)
        self.episode['obst_right_x'].append(self.obstacle.right_x)
        self.episode['obst_bottom_y'].append(self.obstacle.bottom_y)
        self.episode['obst_top_y'].append(self.obstacle.top_y)
        self.episode['obst_y_from_axle'].append(self.obstacle.bottom_y - self.axle_y)
        # eventually store if done
        if self.done:
            self.last_complete_episode = self.episode

    def compute_episode_robustness(self, episode, bool_safety_break=False):
        # compute robustness
        import rtamt
        spec = rtamt.STLSpecification()
        for v, t in zip(self.monitoring_variables, self.monitoring_types):
            spec.declare_var(v, f'{t}')
        # in order to highlight the breaking of safety requirements,
        # the `bool_spec` consider each safety signal as a binary function: values >= 0 till valid, <<0 when violation
        # the `cont_spec` returns a continuous values which is more difficult to interpret
        cont_spec, bool_spec = self.monitoring_specs
        if bool_safety_break:
            spec.spec = bool_spec   # mainly used for evaluation
        else:
            spec.spec = cont_spec
        try:
            spec.parse()
        except rtamt.STLParseException as err:
            return
        # preprocess format, evaluate, post process
        robustness_trace = spec.evaluate(episode)
        return robustness_trace[0][1]

    def reward(self):
        """
        Vanilla Reward - Punish in early termination
        """
        if self.done:
            if self.step_count <= self.max_episode_steps:
                return - 1.0    # early termination: either collision, outside, falldown, battery
            elif abs(self.state[0] - self.x_target) <= self.x_target_tol:
                return + 1.0    # successfully reach the target
            else:
                return 0.0
        else:
            return 0.0

    def reset(self):
        """
        x_init is in [-max_offset,-min_offset] U [min_offset,max_offset]
        obstacles start on same side with a distance from the cart in [min_dist,max_dist]
        state = (x, x_dot, theta, theta_dot, battery, obstacle_left_x, obstacle_right_x, obstacle_height)
        """
        self.n_resets += 1
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(self.state_dim,)).astype(np.float32)
        self.last_state = self.state
        self.steps_beyond_done = None
        self.done = False
        # reset rewards
        self.rew = 0.0
        self.ret = 0.0
        self.x_target_score = 0.0
        self.theta_target_score = 0.0
        self.collision_score = 0.0
        self.falldown_score = 0.0
        self.safety_tot = 0.0
        self.target_tot = 0.0
        self.comfort_tot = 0.0

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
        # sample obstacle parameters: height, width, initial distance from cart
        if self.task == "random_height":
            if self.eval:
                obstacle_height = self.obstacle_min_height if self.n_resets % 2 == 0 else self.obstacle_max_height
            else:
                # in the training, try to keep balanced the number of episodes with obst < or > the feasible height
                if self.np_random.random() <= 0.50:
                    obstacle_height = self.np_random.uniform(low=self.obstacle_min_height,
                                                             high=self.feasible_height)
                else:
                    obstacle_height = self.np_random.uniform(low=self.feasible_height,
                                                             high=self.obstacle_max_height)
        else:
            obstacle_height = self.np_random.uniform(low=self.obstacle_min_height, high=self.obstacle_max_height)
        obstacle_width = self.np_random.uniform(low=self.obstacle_min_width, high=self.obstacle_max_width)
        # distance between cart's initial position (x_init) and the obstacle center
        distance_obst_center_to_cart = self.np_random.uniform(low=self.obstacle_min_dist + obstacle_width / 2,
                                                              high=self.obstacle_max_dist + obstacle_width / 2)
        if self.state[0] > 0:
            left_x = self.state[0] - distance_obst_center_to_cart - obstacle_width / 2.0
        else:
            left_x = self.state[0] + distance_obst_center_to_cart - obstacle_width / 2.0
        axle_y = self.axle_y
        polelen = self.pole_length
        obstacle_y = axle_y + obstacle_height  # this is the bottom y of the obstacle
        self.obstacle = Obstacle(axle_y, polelen, left_x, obstacle_y, obstacle_width, obstacle_height)
        # store obstacle position into the state
        self.state[5] = self.obstacle.left_x
        self.state[6] = self.obstacle.right_x
        self.state[7] = obstacle_height
        # store initial state to check obstacle overcoming
        self.initial_state = np.array(self.state)
        # reset episode
        self.episode = {v: [] for v in self.monitoring_variables}
        return np.array(self.state)

    def set_obstacle_width_height(self, width, height):
        self.obstacle_max_width = width
        self.obstacle_max_height = height

    def overcome_obstacle(self, x):
        """
        Check if the current state is on the opposite side of the obstacle w.r.t. the starting position.
        In particular, we check if the sides are different.
        """
        return self.obstacle.on_left_side(self.initial_state[0]) != self.obstacle.on_left_side(self.state[0])

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
            t, b = -0.1 * scale / 2.0, 0.1 * scale / 2.0
            obst = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.obsttrans = rendering.Transform()
            obst.set_color(*obstacle_color)
            obst.add_attr(self.obsttrans)
            self.viewer.add_geom(obst)

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

        # note: translate obstacle center
        # new center:   x: left_x + obstacle_width
        #               y: cart_center_y + pole_length + obstacle_height/2
        obstx = (self.state[5] + obstacle_width / 2.0) * scale + screen_width / 2.0
        obsty = (self.obstacle.bottom_y + 0.1 / 2.0) * scale
        self.obsttrans.set_translation(obstx, obsty)

        self.label.text = f'safety = {self.safety_tot:.2f}, target = {self.target_tot:.2f}, comfort: {self.comfort_tot:.2f}\n' \
                          f'time: {self.step_count}, reward = {self.rew:.2f}, return = {self.ret:.2f}'
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
