import collections
import math
from typing import Dict, Any

import numpy as np

import gym
from numba import njit

from f110_gym.envs import F110Env
from f110_gym.envs.rendering import EnvRenderer
from reward_shaping.envs.f1tenth.core.utils.track import Track


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


class SingleAgentRaceEnv(F110Env):
    """
    This class implement a base wrapper to the original f1tenth_gym environment tailored on Single-Agent race,
    introducing the following changes:
        - observation and action spaces defined as dictionary with low/high limits of each observable quantities
        - automatic reset of the agent initial position
        - fix rendering issue based on map filepath
    """

    def __init__(self, map_name: str, gui: bool = False, sim_params: Dict[str, Any] = {},
                 actions_conf: Dict[str, Any] = {}, observations_conf: Dict[str, Any] = {},
                 termination_conf: Dict[str, Any] = {},
                 comfortable_speed_limit: float = 7.0,
                 comfortable_steering: float = 0.4189, favourite_lane: int = 0,
                 eval: bool = False, seed: int = None):
        self._track = Track.from_track_name(map_name)
        seed = np.random.randint(0, 1000000) if seed is None else seed
        self.sim_params = self.process_sim_conf(sim_params)
        self.actions_conf = self.process_action_conf(actions_conf)
        self.observations_conf = self.process_obs_conf(observations_conf)
        self.termination_conf = self.process_term_conf(termination_conf)
        #
        super(SingleAgentRaceEnv, self).__init__(map=self._track.filepath, map_ext=self._track.ext,
                                                 params=self.sim_params, num_agents=1, seed=seed)
        self.add_render_callback(render_callback)
        self._scan_size = self.sim.agents[0].scan_simulator.num_beams
        self._scan_range = self.sim.agents[0].scan_simulator.max_range
        # episode
        self.eval = eval
        # task spec
        self.comf_speed_max = comfortable_speed_limit
        self.comf_steering = comfortable_steering
        self.favourite_lane = favourite_lane
        # rendering
        self._gui = False 
        self._render_freq = 10
        self._step = 0
        # state
        self.p0 = 0.0
        self.progress = 0.0
        self.reverse = False

    @property
    def track(self):
        return self._track

    @property
    def observation_space(self):
        """
            scan: lidar data (m)
            pose: x, y, z coordinate (m)
            velocity: linear x velocity (m/s), linear y velocity (m/s), angular velocity (rad/s)
            collision: 0 if not collision, 1 if collision
        """
        obsdict = {}
        l2d_bins = math.ceil(2 * self.observations_conf["max_range"] / self.observations_conf["resolution"])
        obs_spaces = {"scan": gym.spaces.Box(low=0.0, high=self._scan_range, shape=(self._scan_size,)),
                      "lidar_occupancy": gym.spaces.Box(low=0, high=255, dtype=np.uint8, shape=(1, l2d_bins, l2d_bins)),
                      "pose": gym.spaces.Box(low=np.NINF, high=np.PINF, shape=(3,)),
                      "velocity": gym.spaces.Box(low=-5.0, high=20.0, shape=(1,)),
                      "speed_cmd": gym.spaces.Box(low=0.0, high=10.0, shape=(1,)),
                      "steering_cmd": gym.spaces.Box(low=-0.4189, high=0.4189, shape=(1,)),
                      "collision": gym.spaces.Box(low=0.0, high=1.0, shape=(1,)),
                      "reverse": gym.spaces.Box(low=0.0, high=1.0, shape=(1,)),
                      "progress": gym.spaces.Box(low=0.0, high=1.0, shape=(1,)),
                      "progress_meters": gym.spaces.Box(low=0.0, high=np.PINF, shape=(1,)),
                      "lane": gym.spaces.Box(low=0.0, high=1.0, shape=(1,)),
                      }
        for obs, space in obs_spaces.items():
            if obs not in self.observations_conf["types"]:
                continue
            obsdict[obs] = space
        return gym.spaces.Dict(obsdict)

    @property
    def action_space(self):
        """
            steering: desired steering angle (rad)
            speed: desired speed (m/s)
        """
        steering_low, steering_high = self.actions_conf['min_steering'], self.actions_conf['max_steering']
        speed_low, speed_high = self.actions_conf['min_speed'], self.actions_conf['max_speed']
        assert speed_low > 0, "be careful with 0 velocity, it could cause division-by-zero"
        return gym.spaces.Dict({
            "steering": gym.spaces.Box(low=steering_low, high=steering_high, shape=()),
            "speed": gym.spaces.Box(low=speed_low, high=speed_high, shape=())
        })

    @staticmethod
    def _process_conf(default, params):
        conf = default
        for k, v in params.items():
            conf[k] = v
        return conf

    def process_sim_conf(self, sim_params):
        default = {'mu': 1.0489, 'C_Sf': 4.718, 'C_Sr': 5.4562, 'lf': 0.15875, 'lr': 0.17145, 'h': 0.074, 'm': 3.74,
                   'I': 0.04712, 's_min': -0.4189, 's_max': 0.4189, 'sv_min': -3.2, 'sv_max': 3.2,
                   'v_switch': 7.319, 'a_max': 9.51, 'v_min': -5.0, 'v_max': 20.0, 'width': 0.31, 'length': 0.58}
        return self._process_conf(default, sim_params)

    def process_action_conf(self, action_conf):
        default = {"min_steering": -0.4189, "max_steering": 0.4189, "min_speed": -5.0, "max_speed": 10.0}
        return self._process_conf(default, action_conf)

    def process_obs_conf(self, obs_conf):
        default = {
            'types': ['scan', 'pose', 'velocity', 'lidar_occupancy', 'speed_cmd', 'steering_cmd', 'velocity',
                      'progress', 'progress_meters', 'collision', 'reverse', 'lane'],
            'max_range': 10.0, 'resolution': 0.25, 'frame_skip': 4, 'degree_fov': 360}
        return self._process_conf(default, obs_conf)

    def process_term_conf(self, termination_conf):
        default = {"train_max_steps": 1500, "max_lap": 1, "on_collision": True, "train_progress_target": 1.0,
                   "max_steps": 60000}
        return self._process_conf(default, termination_conf)

    def _preprocess_action(self, action: Dict[str, float]):
        assert 'steering' in action and 'speed' in action
        flat_action = np.array(
            [[np.clip(action['steering'], self.actions_conf["min_steering"], self.actions_conf["max_steering"]),
              np.clip(action['speed'], self.actions_conf["min_speed"], self.actions_conf["max_speed"])]])
        assert flat_action.shape == (
            1, 2), f'the actions dict-array conversion returns wrong shape {flat_action.shape}'
        return flat_action

    def l2d_observation(self, observation):
        assert 'scans' in observation
        # params
        degree_fov = self.observations_conf["degree_fov"]
        resolution = self.observations_conf["resolution"]
        l2d_bins = math.ceil(2 * self.observations_conf["max_range"] / self.observations_conf["resolution"])
        # obs
        scan = observation['scans'][0]
        scan_angles = self.sim.agents[0].scan_angles  # assumption: all the lidars are equal in ang. spectrum
        # reduce fow
        mask = abs(scan_angles) <= np.deg2rad(degree_fov / 2.0)  # mask fov: set 1 for angles in fow, 0 for others
        scan = np.where(mask, scan, np.Inf)
        return polar2cartesian(scan, scan_angles, l2d_bins, resolution)

    def prepare_obs(self, old_obs, action):
        assert all([f in old_obs for f in ['scans', 'poses_x', 'poses_y', 'poses_theta',
                                           'linear_vels_x', 'linear_vels_y', 'ang_vels_z',
                                           'collisions']]), f'obs keys are {old_obs.keys()}'
        # Note: the original env returns `scan` values > `max_range`. To keep compatibility wt obs-space, we clip it
        all_obs = {'scan': np.clip(old_obs['scans'][0], 0, self._scan_range),
                   'lidar_occupancy': self.l2d_observation(old_obs),
                   'pose': np.array([old_obs['poses_x'][0], old_obs['poses_y'][0], old_obs['poses_theta'][0]]),
                   'velocity': np.array([old_obs['linear_vels_x'][0]]),
                   'speed_cmd': np.array([action["speed"]]),
                   'steering_cmd': np.array([action["steering"]]),
                   'progress': self.progress,
                   'progress_meters': self.progress * self.track.track_length,
                   'lane': self._track.get_lane(np.array([old_obs['poses_x'][0], old_obs['poses_y'][0]])),
                   'collision': old_obs['collisions'][0],
                   'reverse': self.reverse}
        filtered_obs = {}
        for obs, value in all_obs.items():
            if obs not in self.observations_conf["types"]:
                continue
            filtered_obs[obs] = value

        return filtered_obs

    def prepare_info(self, old_obs, reward, action, old_info):
        assert all([f in old_obs for f in
                    ['lap_times', 'lap_counts', 'collisions', 'linear_vels_x']]), f'obs keys are {old_obs.keys()}'
        assert all([f in action for f in ['steering', 'speed']]), f'action keys are {action.keys()}'
        assert all([f in old_info for f in ['checkpoint_done']]), f'info keys are {old_info.keys()}'
        # target_progress := the space that the car can cover with the comfortable speed (or the max track len for shorter tracks)
        target_progress_mt = min(self.track.track_length,
                                 self.timestep * self.termination_conf["max_steps"] * (
                                         self.actions_conf["min_speed"] + (
                                             self.actions_conf["max_speed"] - self.actions_conf["min_speed"]) / 2))
        info = {'checkpoint_done': old_info['checkpoint_done'][0],
                'lap_time': old_obs['lap_times'][0],
                'lap_count': old_obs['lap_counts'][0],
                'collision': bool(old_obs['collisions'][0]),
                'velocity': old_obs['linear_vels_x'][0],
                'time': self._step * self.timestep,
                'progress': self.progress,
                'progress_meters': self.progress * self.track.track_length,
                'track_length': self.track.track_length,
                'reverse_direction': self.reverse,
                'progress_target_meters': target_progress_mt,
                'action': action,
                'default_reward': reward,
                'comfortable_speed_limit': self.comf_speed_max,
                'comfortable_steering': self.comf_steering,
                'favourite_lane': self.favourite_lane,
                'max_speed': self.actions_conf["max_speed"],
                'max_steering': self.actions_conf["max_steering"],
                'max_steps': self.termination_conf["max_steps"]
                }
        return info

    def update_progress(self, obs):
        new_progress = obs['lap_counts'][0] + self._track.get_progress(
            np.array([obs['poses_x'][0], obs['poses_y'][0]])) - self.p0
        self.reverse = new_progress < self.progress
        self.progress = new_progress

    def check_termination_conditions(self, obs, info):
        done = False
        if self._step > self.termination_conf["max_steps"]:
            done = True
        if not self.eval and self.progress * self.track.track_length > info["progress_target_meters"]:
            done = True
        if self.termination_conf["on_collision"] and info["collision"]:
            done = True
        if info["lap_count"] >= self.termination_conf["max_lap"]:
            done = True
        if self.termination_conf["on_reverse"] and info["reverse_direction"]:
            done = True
        return done

    def step(self, action):
        """
        Note: `step` is used in the `reset` method of the original environment, to initialize the simulators
        For this reason, we cannot rid off the original fields in the observation completely.
        We overcome this by checking if the action is an array (then called in the reset) or a dictionary (otherwise)
        """
        if type(action) == np.ndarray:
            obs, reward, done, info = super().step(action)
        else:
            flat_action = self._preprocess_action(action)
            original_obs, reward, done, original_info = super().step(flat_action)
            self.update_progress(original_obs)
            obs = self.prepare_obs(original_obs, action)
            info = self.prepare_info(original_obs, reward, action, original_info)
            done = self.check_termination_conditions(obs, info)
            info["done"] = done
        if self._gui and self._step % self._render_freq == 0:
            self.render()
        self._step += 1
        return obs, reward, done, info

    def reset(self, mode: str = 'grid'):
        """ reset modes:
                - grid: reset the agent position on the first waypoint
                - random: reset the agent position on a random waypoint
        """
        assert mode in ['grid', 'random']
        if mode == "grid":
            waypoint_id = 0
        elif mode == "random":
            waypoint_id = np.random.randint(self._track.centerline.shape[0] - 1)
        else:
            raise NotImplementedError(f"reset mode {mode} not implemented")
        # compute x, y, theta
        assert 0 <= waypoint_id < self._track.centerline.shape[0] - 1
        wp, next_wp = self._track.centerline[waypoint_id], self._track.centerline[waypoint_id + 1]
        theta = np.arctan2(next_wp[1] - wp[1], next_wp[0] - wp[0])
        pose = [wp[0], wp[1], theta]
        # call original method
        original_obs, reward, done, original_info = super().reset(poses=np.array([pose]))
        self.p0 = self._track.get_progress(np.array([original_obs['poses_x'][0], original_obs['poses_y'][0]]))
        self.progress = 0.0
        self.reverse = False
        self._step = 0
        obs = self.prepare_obs(original_obs, {"steering": 0.0, "speed": 0.0})
        return obs

    def render(self, mode='human', **kwargs):
        assert mode in ["human", "rgb_array"]
        WINDOW_W, WINDOW_H = 1000, 800
        if self.renderer is None:
            # first call, initialize everything
            F110Env.renderer = EnvRenderer(WINDOW_W, WINDOW_H)
            F110Env.renderer.update_map(self._track.filepath, self._track.ext)

        if mode == "human":
            super(SingleAgentRaceEnv, self).render(mode)
        elif mode == "rgb_array":
            super(SingleAgentRaceEnv, self).render("human")
            from pyglet.gl import GLubyte
            buffer = (GLubyte * (3 * F110Env.renderer.width * F110Env.renderer.height))(0)
            from pyglet.gl import glReadPixels
            from pyglet.gl import GL_RGB
            from pyglet.gl import GL_UNSIGNED_BYTE
            glReadPixels(0, 0, self.renderer.width, self.renderer.height, GL_RGB, GL_UNSIGNED_BYTE, buffer)

            # Use PIL to convert raw RGB buffer and flip the right way up
            from PIL import Image
            image = Image.frombytes(mode="RGB", size=(self.renderer.width, self.renderer.height), data=buffer)
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            image = image.resize((400, 320), Image.ANTIALIAS)
            return np.array(image)


def render_callback(env_renderer):
    # custom extra drawing function
    return
    e = env_renderer

    # update camera to follow car
    x = e.cars[0].vertices[::2]
    y = e.cars[0].vertices[1::2]
    top, bottom, left, right = max(y), min(y), min(x), max(x)
    e.left = left - 400
    e.right = right + 400
    e.top = top + 400
    e.bottom = bottom - 400


if __name__ == "__main__":
    env = SingleAgentRaceEnv("Catalunya")
    for i in range(2):
        print(f"episode {i + 1}")
        obs = env.reset(mode='random')
        for j in range(500):
            obs, reward, done, info = env.step({'steering': 0.0, 'speed': 2.0})
            env.render()
        print()
    # check env
    try:
        from stable_baselines3.common.env_checker import check_env

        check_env(env)
        print("[Result] env ok")
    except Exception as ex:
        print("[Result] env not compliant wt openai-gym standard")
        print(ex)
