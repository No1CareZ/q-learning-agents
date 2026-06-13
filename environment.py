"""Environment"""

import numpy as np
import math
import random


class Environment:
    """."""

    def __init__(self):
        # Map
        self.x_min, self.x_max = 0.0, 24.0
        self.y_min, self.y_max = 0.0, 12.0

        # Charge
        self.station_center = np.array([21.5, 8.5])
        self.station_radius = 0.9
        self.charge_power = 0.025

        # Start
        self.start_mean = np.array([3.5, 6.0])
        self.start_std = 1.8
        self.start_clip = {"x": (1.0, 7.0), "y": (2.0, 10.0)}

        # Wind
        self.wind_range = (-0.11, 0.11)
        self.wind_update_interval = 15

        # Blocks
        self.obstacles = [
            [(10.0, 2.0), (11.5, 5.5)],
            [(13.0, 7.0), (14.5, 10.0)],
            [(16.5, 3.5), (18.0, 6.5)],
            [(19.0, 9.0), (20.5, 11.5)],
        ]

        # States
        self.delta_x_bins = 14
        self.delta_y_bins = 14
        self.battery_bins = 12
        self.speed_values = [0.10, 0.25, 0.40, 0.60]
        self.num_speeds = len(self.speed_values)

        # Deltas (some QOL)
        self.delta_x_min = self.x_min - self.station_center[0]
        self.delta_x_max = self.x_max - self.station_center[0]
        self.delta_y_min = self.y_min - self.station_center[1]
        self.delta_y_max = self.y_max - self.station_center[1]

        # Bin sizes
        self.delta_x_bin_size = (
            self.delta_x_max - self.delta_x_min
        ) / self.delta_x_bins
        self.delta_y_bin_size = (
            self.delta_y_max - self.delta_y_min
        ) / self.delta_y_bins
        self.battery_bin_size = 1.0 / self.battery_bins

        # Actions
        self.directions = [0, math.pi / 2, math.pi, 3 * math.pi / 2]
        self.num_directions = len(self.directions)
        self.num_actions = self.num_directions * self.num_speeds

        self.action_to_dir_speed = {}
        # {
        #     action_idx : (dir_idx, speed_idx)
        # }
        for a in range(self.num_actions):
            dir_idx = a // self.num_speeds
            speed_idx = a % self.num_speeds
            self.action_to_dir_speed[a] = (dir_idx, speed_idx)

        # Charger constants
        self.battery_v_coeff = 0.018
        self.battery_w_coeff = 0.003
        self.battery_v_abs_coeff = 0.002

        # End
        self.success_battery_threshold = 0.90  # 0.90 # 0.77
        self.failure_battery_threshold = 0.04
        self.max_steps = 200
        self.success_time_threshold = 140

        # Rewards (yay!)
        self.success_reward = 90.0
        self.failure_reward = -170.0
        self.default_reward = -1.0
        self.speed_penalty_coeff = 0.030
        self.charge_bonus_coeff = 0.12 * self.charge_power

        # Counters
        self.current_step = 0
        self.current_wind = None
        self.state = None

    def reset(self):
        """Reset."""
        self.current_step = 0

        # Positioning
        x0 = np.random.normal(self.start_mean[0], self.start_std)
        y0 = np.random.normal(self.start_mean[1], self.start_std)
        x0 = np.clip(x0, self.start_clip["x"][0], self.start_clip["x"][1])
        y0 = np.clip(y0, self.start_clip["y"][0], self.start_clip["y"][1])

        # Speed
        v0 = self.speed_values[0]

        # Wind
        self.current_wind = self._generate_wind()

        # Charge
        b0 = 1.0

        # State
        self.state = np.array([x0, y0, b0, v0])

        if self._check_collision(x0, y0):  # Check if we crush from start
            x0, y0 = self._project_to_boundary(x0, y0)
            self.state[0], self.state[1] = x0, y0

        return self._bin_state(self.state)

    def step(self, action_idx):
        """Make 1 step."""
        # Drone action
        dir_idx, speed_idx = self.action_to_dir_speed[action_idx]
        theta = self.directions[dir_idx]
        target_speed = self.speed_values[speed_idx]

        # State
        x, y, b, v = self.state
        wx, wy = self.current_wind

        # Wind
        if self.current_step % self.wind_update_interval == 0:
            self.current_wind = self._generate_wind()
            wx, wy = self.current_wind

        # Next position
        dx = target_speed * math.cos(theta) + wx
        dy = target_speed * math.sin(theta) + wy
        new_x = x + dx
        new_y = y + dy

        # Clip
        clip_debuff = 0.0
        if (
            new_x > self.x_max
            or new_x < self.x_min
            or new_y > self.y_max
            or new_y < self.y_min
        ):
            new_x = np.clip(new_x, self.x_min, self.x_max)
            new_y = np.clip(new_y, self.y_min, self.y_max)
            clip_debuff = 2.0

        # Collision
        collision_check = False
        collision_debuff = 0.0
        if self._check_collision(new_x, new_y):
            new_x, new_y = self._project_to_boundary(new_x, new_y)
            collision_check = True
            collision_debuff = 2.0

        # Drain charge
        battery_drain = (
            self.battery_v_coeff * target_speed**2
            + self.battery_w_coeff * (wx**2 + wy**2)
            + self.battery_v_abs_coeff * abs(target_speed)
        )

        # Charge if need
        dist_to_station = np.linalg.norm(
            np.array([new_x, new_y]) - self.station_center
        )  # Distance based on euclidian distance (I hate this line btw)
        in_station = dist_to_station < self.station_radius

        battery_charge = 0
        if in_station:
            battery_charge = self.charge_power
        new_b = min(1.0, b - battery_drain + battery_charge)  # Clip charge!!!!

        # Speed
        new_v = target_speed

        # New state (for agent)
        new_state = np.array([new_x, new_y, new_b, new_v])
        new_discrete_state = self._bin_state(new_state)

        # Rewards
        success = False
        failure = False

        if (
            in_station
            and new_b > self.success_battery_threshold
            and self.current_step < self.success_time_threshold
        ):
            success = True

        if (
            new_b < self.failure_battery_threshold
            or self.current_step >= self.max_steps - 1
        ):
            failure = True

        if success:
            reward = self.success_reward
        elif failure:
            reward = self.failure_reward
        else:
            reward = (
                self.default_reward
                - self.speed_penalty_coeff * target_speed**2
                + (self.charge_bonus_coeff * in_station)  # if in_station else 0.0)
            )

        reward -= collision_debuff
        reward -= clip_debuff

        # Update state
        self.state = new_state
        self.current_step += 1
        done = success or failure

        # Log
        info = {
            "position": (new_x, new_y),
            "battery": new_b,
            "speed": new_v,
            "target_speed": target_speed,
            "direction": theta,
            "wind": (wx, wy),
            "in_station": in_station,
            "collision": collision_check,
            "success": success,
            "failure": failure,
            "distance_to_station": dist_to_station,
        }

        return new_discrete_state, reward, done, info

    def _generate_wind(self):
        """Wind."""
        wx = random.uniform(self.wind_range[0], self.wind_range[1])
        wy = random.uniform(self.wind_range[0], self.wind_range[1])
        return np.array([wx, wy])

    def _check_collision(self, x, y):
        """Did we hit the wall?"""
        point = np.array([x, y])
        for obs in self.obstacles:
            obs_min = np.array(obs[0])
            obs_max = np.array(obs[1])
            if (point >= obs_min).all() and (point <= obs_max).all():
                return True
        return False

    def _project_to_boundary(self, x, y):
        """Return back if OfB."""
        point = np.array([x, y])
        for obs in self.obstacles:
            obs_min = np.array(obs[0])
            obs_max = np.array(obs[1])
            if (point >= obs_min).all() and (point <= obs_max).all():
                distances = {
                    "left": x - obs_min[0],
                    "right": obs_max[0] - x,
                    "bottom": y - obs_min[1],
                    "top": obs_max[1] - y,
                }
                min_dist = min(distances.values())
                if min_dist == distances["left"]:
                    return obs_min[0] - 0.01, y  # 0.01 casue I am tired
                elif min_dist == distances["right"]:
                    return obs_max[0] + 0.01, y
                elif min_dist == distances["bottom"]:
                    return x, obs_min[1] - 0.01
                else:
                    return x, obs_max[1] + 0.01
        return x, y

    def _bin_state(self, state):
        """Use given bins to form indexes."""
        x, y, b, v = state

        delta_x = x - self.station_center[0]
        delta_y = y - self.station_center[1]

        delta_x_idx = int((delta_x - self.delta_x_min) / self.delta_x_bin_size)
        delta_x_idx = np.clip(delta_x_idx, 0, self.delta_x_bins - 1)

        delta_y_idx = int((delta_y - self.delta_y_min) / self.delta_y_bin_size)
        delta_y_idx = np.clip(delta_y_idx, 0, self.delta_y_bins - 1)

        b_idx = int(b / self.battery_bin_size)
        b_idx = np.clip(b_idx, 0, self.battery_bins - 1)

        v_idx = min(
            range(self.num_speeds), key=lambda i: abs(v - self.speed_values[i])
        )  # line too long

        return (delta_x_idx, delta_y_idx, b_idx, v_idx)

    def get_state_size(self):
        """Return state size."""
        return (
            self.delta_x_bins * self.delta_y_bins * self.battery_bins * self.num_speeds
        )

    def get_action_size(self):
        """Return action size."""
        return self.num_actions
