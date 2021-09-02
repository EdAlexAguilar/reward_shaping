"""
Rocket trajectory optimization is a classic topic in Optimal Control.

According to Pontryagin's maximum principle it's optimal to fire engine full throttle or
turn it off. That's the reason this environment is OK to have discreet actions (engine on or off).

The landing pad is always at coordinates (0,0). The coordinates are the first two numbers in the state vector.
Reward for moving from the top of the screen to the landing pad and zero speed is about 100..140 points.
If the lander moves away from the landing pad it loses reward. The episode finishes if the lander crashes or
comes to rest, receiving an additional -100 or +100 points. Each leg with ground contact is +10 points.
Firing the main engine is -0.3 points each frame. Firing the side engine is -0.03 points each frame.
Solved is 200 points.

Landing outside the landing pad is possible. Fuel is infinite, so an agent can learn to fly and then land
on its first attempt. Please see the source code for details.

To see a heuristic landing, run:

python gym/envs/box2d/lunar_lander.py

To play yourself, run:

python examples/agents/keyboard_agent.py LunarLander-v2

Created by Oleg Klimov. Licensed on the same terms as the rest of OpenAI Gym.
"""

import sys, math
from typing import Dict

import numpy as np

import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener)

import gym
from gym import spaces
from gym.spaces import Box
from gym.utils import seeding, EzPickle

SCALE = 30.0  # affects how fast-paced the game is, forces should be adjusted as well
MAIN_ENGINE_POWER = 13.0
SIDE_ENGINE_POWER = 0.6

INITIAL_RANDOM = 1000.0  # Set 1500 to make game harder

LANDER_POLY = [
    (-14, +17), (-17, 0), (-17, -10),
    (+17, -10), (+17, 0), (+14, +17)
]
LEG_AWAY = 20
LEG_DOWN = 18
LEG_W, LEG_H = 2, 8
LEG_SPRING_TORQUE = 40

SIDE_ENGINE_HEIGHT = 14.0
SIDE_ENGINE_AWAY = 12.0

VIEWPORT_W = 600
VIEWPORT_H = 400


class ContactDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env

    def BeginContact(self, contact):
        if self.env.lander == contact.fixtureA.body or self.env.lander == contact.fixtureB.body:
            self.env.game_over = True
        for i in range(2):
            if self.env.legs[i] in [contact.fixtureA.body, contact.fixtureB.body]:
                self.env.legs[i].ground_contact = True

    def EndContact(self, contact):
        for i in range(2):
            if self.env.legs[i] in [contact.fixtureA.body, contact.fixtureB.body]:
                self.env.legs[i].ground_contact = False


class LunarLander(gym.Env, EzPickle):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    continuous = False

    def __init__(self, task='land', FPS=50, x_low_limit=0.0, x_high_limit=20.0, initial_x_offset=2.0, fuel_usage=0.005,
                 x_target=0.0, y_target=0.0, halfwidth_landing_area=1.0,
                 theta_limit=1, theta_dot_limit=0.5,
                 obstacle_lowleft_x=10.0, obstacle_lowleft_y=7.0, obstacle_width=2.0, obstacle_height=0.5,
                 eval=False, seed=0):
        EzPickle.__init__(self)
        # env params
        self.task = task
        self.x_low_limit, self.x_high_limit = x_low_limit, x_high_limit
        self.initial_x_offset = initial_x_offset
        self.x_target, self.y_target = x_target, y_target
        self.halfwidth_landing_area = halfwidth_landing_area
        self.discrete_fuel_usage = fuel_usage
        self.eval = eval

        self.seed(seed)
        self.viewer = None
        self.obstacle_vertices = (
            obstacle_lowleft_x, obstacle_lowleft_y, *[coord + size for coord, size in
                                                      zip((obstacle_lowleft_x, obstacle_lowleft_y),
                                                          (obstacle_width, obstacle_height))])
        self.FPS = FPS
        self.world = Box2D.b2World()
        self.moon = None
        self.obstacle = None
        self.lander = None
        self.particles = []
        self.prev_reward = None
        self.theta_limit = theta_limit
        self.theta_dot_limit = theta_dot_limit
        # useful range is -1 .. +1, but spikes can be higher
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(14,), dtype=np.float32)
        self.observation_space = gym.spaces.Dict(dict(
            x=Box(low=-np.Inf, high=np.Inf, shape=(1,), dtype=np.float32),
            y=Box(low=-np.Inf, high=np.Inf, shape=(1,), dtype=np.float32),
            horizontal_speed=Box(low=-np.Inf, high=np.Inf, shape=(1,), dtype=np.float32),
            vertical_speed=Box(low=-np.Inf, high=np.Inf, shape=(1,), dtype=np.float32),
            angle=Box(low=-np.Inf, high=np.Inf, shape=(1,), dtype=np.float32),
            angle_speed=Box(low=-np.Inf, high=np.Inf, shape=(1,), dtype=np.float32),
            ground_contact_leg0=Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            ground_contact_leg1=Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            obstacle_left=Box(low=-np.Inf, high=np.Inf, shape=(1,), dtype=np.float32),
            obstacle_right=Box(low=-np.Inf, high=np.Inf, shape=(1,), dtype=np.float32),
            obstacle_top=Box(low=-np.Inf, high=np.Inf, shape=(1,), dtype=np.float32),
            obstacle_bottom=Box(low=-np.Inf, high=np.Inf, shape=(1,), dtype=np.float32),
            fuel=Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            collision=Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
        ))

        if self.continuous:
            # Action is two floats [main engine, left-right engines].
            # Main engine: -1..0 off, 0..+1 throttle from 50% to 100% power. Engine can't work with less than 50% power.
            # Left-right:  -1.0..-0.5 fire left engine, +0.5..+1.0 fire right engine, -0.5..0.5 off
            self.action_space = spaces.Box(-1, +1, (2,), dtype=np.float32)
        else:
            # Nop, fire left engine, main engine, right engine
            self.action_space = spaces.Discrete(4)

        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _destroy(self):
        if not self.moon: return
        self.world.contactListener = None
        self._clean_particles(True)
        self.world.DestroyBody(self.moon)
        self.moon = None
        self.world.DestroyBody(self.obstacle)
        self.obstacle = None
        self.world.DestroyBody(self.lander)
        self.lander = None
        self.world.DestroyBody(self.legs[0])
        self.world.DestroyBody(self.legs[1])

    def reset(self):
        self._destroy()
        self.world.contactListener_keepref = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_keepref
        self.game_over = False
        self.prev_shaping = None

        W = VIEWPORT_W / SCALE
        H = VIEWPORT_H / SCALE

        self.step_count = 0
        self.fuel = 1

        # terrain
        CHUNKS = 11
        height = self.np_random.uniform(0, H / 2, size=(CHUNKS + 1,))
        chunk_x = [W / (CHUNKS - 1) * i for i in range(CHUNKS)]
        self.helipad_x1 = chunk_x[CHUNKS // 2 - 1]
        self.helipad_x2 = chunk_x[CHUNKS // 2 + 1]
        self.helipad_y = H / 4
        height[CHUNKS // 2 - 2] = self.helipad_y
        height[CHUNKS // 2 - 1] = self.helipad_y
        height[CHUNKS // 2 + 0] = self.helipad_y
        height[CHUNKS // 2 + 1] = self.helipad_y
        height[CHUNKS // 2 + 2] = self.helipad_y
        smooth_y = [0.33 * (height[i - 1] + height[i + 0] + height[i + 1]) for i in range(CHUNKS)]

        self.moon = self.world.CreateStaticBody(shapes=edgeShape(vertices=[(0, 0), (W, 0)]))
        self.obstacle = self.world.CreateStaticBody(
            shapes=edgeShape(vertices=[self.obstacle_vertices[:2], self.obstacle_vertices[2:]]))
        x1, y1, x2, y2 = self.obstacle_vertices
        obstacle_edges = [
            [(x1, y1), (x1, y2)],
            [(x1, y2), (x2, y2)],
            [(x2, y2), (x2, y1)],
            [(x2, y1), (x1, y1)]
        ]

        for vertices in obstacle_edges:
            self.obstacle.CreateEdgeFixture(
                vertices=vertices,
                density=0,
                friction=0.1)

        self.sky_polys = []
        for i in range(CHUNKS - 1):
            p1 = (chunk_x[i], smooth_y[i])
            p2 = (chunk_x[i + 1], smooth_y[i + 1])
            self.moon.CreateEdgeFixture(
                vertices=[p1, p2],
                density=0,
                friction=0.1)
            self.sky_polys.append([p1, p2, (p2[0], H), (p1[0], H)])

        self.moon.color1 = (0.0, 0.0, 0.0)
        self.moon.color2 = (0.0, 0.0, 0.0)
        self.obstacle.color1 = (0.5, 0, 0)
        self.obstacle.color2 = (0.5, 0, 0)

        initial_x = VIEWPORT_W / SCALE / 2 + 2 * self.initial_x_offset * (np.random.random() - .5)
        initial_y = VIEWPORT_H / SCALE

        self.lander = self.world.CreateDynamicBody(
            position=(initial_x, initial_y),
            angle=0.0,
            fixtures=fixtureDef(
                shape=polygonShape(vertices=[(x / SCALE, y / SCALE) for x, y in LANDER_POLY]),
                density=5.0,
                friction=0.1,
                categoryBits=0x0010,
                maskBits=0x001,  # collide only with ground
                restitution=0.0)  # 0.99 bouncy
        )
        self.lander.color1 = (0.5, 0.4, 0.9)
        self.lander.color2 = (0.3, 0.3, 0.5)
        self.lander.ApplyForceToCenter((
            self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM),
            self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM)
        ), True)

        self.legs = []
        for i in [-1, +1]:
            leg = self.world.CreateDynamicBody(
                position=(initial_x - i * LEG_AWAY / SCALE, initial_y),
                angle=(i * 0.05),
                fixtures=fixtureDef(
                    shape=polygonShape(box=(LEG_W / SCALE, LEG_H / SCALE)),
                    density=1.0,
                    restitution=0.0,
                    categoryBits=0x0020,
                    maskBits=0x001)
            )
            leg.ground_contact = False
            leg.color1 = (0.5, 0.4, 0.9)
            leg.color2 = (0.3, 0.3, 0.5)
            rjd = revoluteJointDef(
                bodyA=self.lander,
                bodyB=leg,
                localAnchorA=(0, 0),
                localAnchorB=(i * LEG_AWAY / SCALE, LEG_DOWN / SCALE),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=LEG_SPRING_TORQUE,
                motorSpeed=+0.3 * i  # low enough not to jump back into the sky
            )
            if i == -1:
                rjd.lowerAngle = +0.9 - 0.5  # The most esoteric numbers here, angled legs have freedom to travel within
                rjd.upperAngle = +0.9
            else:
                rjd.lowerAngle = -0.9
                rjd.upperAngle = -0.9 + 0.5
            leg.joint = self.world.CreateJoint(rjd)
            self.legs.append(leg)

        self.drawlist = [self.lander, self.obstacle] + self.legs

        return self.step(np.array([0, 0]) if self.continuous else 0)[0]

    def _create_particle(self, mass, x, y, ttl):
        p = self.world.CreateDynamicBody(
            position=(x, y),
            angle=0.0,
            fixtures=fixtureDef(
                shape=circleShape(radius=2 / SCALE, pos=(0, 0)),
                density=mass,
                friction=0.1,
                categoryBits=0x0100,
                maskBits=0x001,  # collide only with ground
                restitution=0.3)
        )
        p.ttl = ttl
        self.particles.append(p)
        self._clean_particles(False)
        return p

    def _clean_particles(self, all):
        while self.particles and (all or self.particles[0].ttl < 0):
            self.world.DestroyBody(self.particles.pop(0))

    def step(self, action):
        self.step_count += 1
        if self.continuous:
            action = np.clip(action, -1, +1).astype(np.float32)
        else:
            assert self.action_space.contains(action), "%r (%s) invalid " % (action, type(action))

        # Engines
        tip = (math.sin(self.lander.angle), math.cos(self.lander.angle))
        side = (-tip[1], tip[0])
        dispersion = [self.np_random.uniform(-1.0, +1.0) / SCALE for _ in range(2)]

        m_power = 0.0
        if (self.continuous and action[0] > 0.0) or (not self.continuous and action == 2):
            # Main engine
            if self.continuous:
                m_power = (np.clip(action[0], 0.0, 1.0) + 1.0) * 0.5  # 0.5..1.0
                assert m_power >= 0.5 and m_power <= 1.0
            else:
                m_power = 1.0
                self.fuel -= self.discrete_fuel_usage
            ox = (tip[0] * (4 / SCALE + 2 * dispersion[0]) +
                  side[0] * dispersion[1])  # 4 is move a bit downwards, +-2 for randomness
            oy = -tip[1] * (4 / SCALE + 2 * dispersion[0]) - side[1] * dispersion[1]
            impulse_pos = (self.lander.position[0] + ox, self.lander.position[1] + oy)
            p = self._create_particle(3.5,  # 3.5 is here to make particle speed adequate
                                      impulse_pos[0],
                                      impulse_pos[1],
                                      m_power)  # particles are just a decoration
            p.ApplyLinearImpulse((ox * MAIN_ENGINE_POWER * m_power, oy * MAIN_ENGINE_POWER * m_power),
                                 impulse_pos,
                                 True)
            self.lander.ApplyLinearImpulse((-ox * MAIN_ENGINE_POWER * m_power, -oy * MAIN_ENGINE_POWER * m_power),
                                           impulse_pos,
                                           True)

        s_power = 0.0
        if (self.continuous and np.abs(action[1]) > 0.5) or (not self.continuous and action in [1, 3]):
            # Orientation engines
            if self.continuous:
                direction = np.sign(action[1])
                s_power = np.clip(np.abs(action[1]), 0.5, 1.0)
                assert s_power >= 0.5 and s_power <= 1.0
            else:
                direction = action - 2
                s_power = 1.0
                self.fuel -= self.discrete_fuel_usage
            ox = tip[0] * dispersion[0] + side[0] * (3 * dispersion[1] + direction * SIDE_ENGINE_AWAY / SCALE)
            oy = -tip[1] * dispersion[0] - side[1] * (3 * dispersion[1] + direction * SIDE_ENGINE_AWAY / SCALE)
            impulse_pos = (self.lander.position[0] + ox - tip[0] * 17 / SCALE,
                           self.lander.position[1] + oy + tip[1] * SIDE_ENGINE_HEIGHT / SCALE)
            p = self._create_particle(0.7, impulse_pos[0], impulse_pos[1], s_power)
            p.ApplyLinearImpulse((ox * SIDE_ENGINE_POWER * s_power, oy * SIDE_ENGINE_POWER * s_power),
                                 impulse_pos
                                 , True)
            self.lander.ApplyLinearImpulse((-ox * SIDE_ENGINE_POWER * s_power, -oy * SIDE_ENGINE_POWER * s_power),
                                           impulse_pos,
                                           True)

        self.world.Step(1.0 / self.FPS, 6 * 30, 2 * 30)

        pos = self.lander.position
        vel = self.lander.linearVelocity
        # normalize obstacle positioning
        obstacle_botleft_x, obstacle_botleft_y = self.obstacle_vertices[:2]
        obstacle_topright_x, obstacle_topright_y = self.obstacle_vertices[2:]
        obstacle_botleft_x = (obstacle_botleft_x - VIEWPORT_W / SCALE / 2) / (VIEWPORT_W / SCALE / 2)
        obstacle_botleft_y = (obstacle_botleft_y - (self.helipad_y + LEG_DOWN / SCALE)) / (VIEWPORT_H / SCALE / 2)
        obstacle_topright_x = (obstacle_topright_x - VIEWPORT_W / SCALE / 2) / (VIEWPORT_W / SCALE / 2)
        obstacle_topright_y = (obstacle_topright_y - (self.helipad_y + LEG_DOWN / SCALE)) / (VIEWPORT_H / SCALE / 2)

        state = {
            "x": (pos.x - VIEWPORT_W / SCALE / 2) / (VIEWPORT_W / SCALE / 2),
            "y": (pos.y - (self.helipad_y + LEG_DOWN / SCALE)) / (VIEWPORT_H / SCALE / 2),
            "horizontal_speed": vel.x * (VIEWPORT_W / SCALE / 2) / self.FPS,
            "vertical_speed": vel.y * (VIEWPORT_H / SCALE / 2) / self.FPS,
            "angle": self.lander.angle,
            "angle_speed": 20.0 * self.lander.angularVelocity / self.FPS,
            "ground_contact_leg0": 1.0 if self.legs[0].ground_contact else 0.0,
            "ground_contact_leg1": 1.0 if self.legs[1].ground_contact else 0.0,
            "obstacle_left": obstacle_botleft_x,
            "obstacle_right": obstacle_topright_x,
            "obstacle_top": obstacle_topright_y,
            "obstacle_bottom": obstacle_botleft_y,
            "fuel": self.fuel,
            "collision": 1.0 if self.game_over else 0.0
        }

        reward = 0
        shaping = \
            - 100 * np.sqrt(state["x"] * state["x"] + state["y"] * state["y"]) \
            - 100 * np.sqrt(state["horizontal_speed"] * state["horizontal_speed"] + state["vertical_speed"] * state[
                "vertical_speed"]) \
            - 100 * abs(state["angle"]) + 10 * state["ground_contact_leg0"] + 10 * state[
                "ground_contact_leg1"]  # And ten points for legs contact, the idea is if you
        # lose contact again after landing, you get negative reward
        if self.prev_shaping is not None:
            reward = shaping - self.prev_shaping
        self.prev_shaping = shaping

        reward -= m_power * 0.30  # less fuel spent is better, about -30 for heuristic landing
        reward -= s_power * 0.03

        done = False
        if self.game_over or abs(state["x"]) >= 1.0 or self.fuel <= 0:
            done = True
            reward = -100
        if not self.lander.awake:
            done = True
            reward = +100

        info = {"time": self.step_count,
                "FPS": self.FPS,
                "theta_limit": self.theta_limit,
                "theta_dot_limit": self.theta_dot_limit,
                "x_target": self.x_target,
                "y_target": self.y_target,
                "x_low_limit": self.x_low_limit,
                "x_high_limit": self.x_high_limit,
                "obstacle_vertices": self.obstacle_vertices,
                "fuel": self.fuel,
                "half_width": VIEWPORT_W / SCALE / 2,
                "halfwidth_landing_area": self.halfwidth_landing_area,
                "collision": self.game_over,
                "default_reward": reward}

        return state, reward, done, info

    @property
    def state(self) -> Dict:
        return

    def render(self, mode='human', info={}):
        from gym.envs.classic_control import rendering
        if self.viewer is None:
            self.viewer = rendering.Viewer(VIEWPORT_W, VIEWPORT_H)
            self.viewer.set_bounds(0, VIEWPORT_W / SCALE, 0, VIEWPORT_H / SCALE)

        for obj in self.particles:
            obj.ttl -= 0.15
            obj.color1 = (max(0.2, 0.2 + obj.ttl), max(0.2, 0.5 * obj.ttl), max(0.2, 0.5 * obj.ttl))
            obj.color2 = (max(0.2, 0.2 + obj.ttl), max(0.2, 0.5 * obj.ttl), max(0.2, 0.5 * obj.ttl))

        self._clean_particles(False)

        for p in self.sky_polys:
            self.viewer.draw_polygon(p, color=(0, 0, 0))

        for obj in self.particles + self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                if type(f.shape) is circleShape:
                    t = rendering.Transform(translation=trans * f.shape.pos)
                    self.viewer.draw_circle(f.shape.radius, 20, color=obj.color1).add_attr(t)
                    self.viewer.draw_circle(f.shape.radius, 20, color=obj.color2, filled=False, linewidth=2).add_attr(t)
                else:
                    path = [trans * v for v in f.shape.vertices]
                    self.viewer.draw_polygon(path, color=obj.color1)
                    path.append(path[0])
                    self.viewer.draw_polyline(path, color=obj.color2, linewidth=2)

        for x in [self.helipad_x1, self.helipad_x2]:
            flagy1 = self.helipad_y
            flagy2 = flagy1 + 50 / SCALE
            self.viewer.draw_polyline([(x, flagy1), (x, flagy2)], color=(1, 1, 1))
            self.viewer.draw_polygon([(x, flagy2), (x, flagy2 - 10 / SCALE), (x + 25 / SCALE, flagy2 - 5 / SCALE)],
                                     color=(0.8, 0.8, 0))

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


class LunarLanderContinuous(LunarLander):
    continuous = True


def heuristic(env, s):
    """
    The heuristic for
    1. Testing
    2. Demonstration rollout.

    Args:
        env: The environment
        s (list): The state. Attributes:
                  s[0] is the horizontal coordinate
                  s[1] is the vertical coordinate
                  s[2] is the horizontal speed
                  s[3] is the vertical speed
                  s[4] is the angle
                  s[5] is the angular speed
                  s[6] 1 if first leg has contact, else 0
                  s[7] 1 if second leg has contact, else 0
    returns:
         a: The heuristic to be fed into the step function defined above to determine the next step and reward.
    """

    angle_targ = s[0] * 0.5 + s[2] * 1.0  # angle should point towards center
    if angle_targ > 0.4: angle_targ = 0.4  # more than 0.4 radians (22 degrees) is bad
    if angle_targ < -0.4: angle_targ = -0.4
    hover_targ = 0.55 * np.abs(s[0])  # target y should be proportional to horizontal offset

    angle_todo = (angle_targ - s[4]) * 0.5 - (s[5]) * 1.0
    hover_todo = (hover_targ - s[1]) * 0.5 - (s[3]) * 0.5

    if s[6] or s[7]:  # legs have contact
        angle_todo = 0
        hover_todo = -(s[3]) * 0.5  # override to reduce fall speed, that's all we need after contact

    if env.continuous:
        a = np.array([hover_todo * 20 - 1, -angle_todo * 20])
        a = np.clip(a, -1, +1)
    else:
        a = 0
        if hover_todo > np.abs(angle_todo) and hover_todo > 0.05:
            a = 2
        elif angle_todo < -0.05:
            a = 3
        elif angle_todo > +0.05:
            a = 1
    return a


def demo_heuristic_lander(env, seed=None, render=False):
    def _convert_dict_to_array(state):
        vars = ["x", "y", "horizontal_speed", "vertical_speed",
                "angle", "angle_speed", "ground_contact_leg0", "ground_contact_leg1",
                "obstacle_left", "obstacle_right", "obstacle_top", "obstacle_bottom",
                "fuel", "collision"]
        return np.array([state[k] for k in vars])

    env.seed(seed)
    for _ in range(10):
        total_reward = 0
        steps = 0
        s_dict = env.reset()
        while True:
            s = _convert_dict_to_array(s_dict)
            a = heuristic(env, s)
            s_dict, r, done, info = env.step(a)
            total_reward += r

            if render:
                still_open = env.render()
                if still_open == False: break

            if steps % 20 == 0 or done:
                print("observations:", " ".join(["{:+0.2f}".format(x) for x in s]))
                print("step {} total_reward {:+0.2f}".format(steps, total_reward))
            steps += 1
            if done:
                break


if __name__ == '__main__':
    demo_heuristic_lander(LunarLander(), render=True)
