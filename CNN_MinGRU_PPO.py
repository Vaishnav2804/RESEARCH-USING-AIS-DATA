# Standard library imports
import csv
from collections import defaultdict
from datetime import datetime, timedelta
import math # Import math for pi
import os
# Third-party library imports
import geopandas as gpd
import gymnasium as gym
from gymnasium import spaces
import h3
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import torch.nn as nn
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback
import multiprocessing as mp
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.utils import get_linear_fn
from stable_baselines3.common.utils import get_schedule_fn
from stable_baselines3.common.callbacks import BaseCallback
# Local application/specific imports
from minGRU_pytorch import minGRU
import geopandas as gpd
import contextily as ctx
from shapely.geometry import Polygon
import matplotlib
from torch.optim import AdamW
from stable_baselines3.common.policies import ActorCriticPolicy
from torch.optim import AdamW
from torch.nn import GELU
from stable_baselines3.common.vec_env import DummyVecEnv
from tqdm import tqdm
import time
import sys

matplotlib.use("Agg")  # ✅ headless backend (important!)

# --- Constants and Configuration ---
# Define common constants used in the environment
KNOTS_TO_MPS = 0.514444
MIN_SPEED_KNOTS = 8.0
MAX_SPEED_KNOTS = 22.0
NUM_SPEED_LEVELS = 5
H3_RESOLUTION = 6
DEFAULT_WIND_THRESHOLD = 10.0 # m/s
DEFAULT_HORIZON_MULTIPLIER = 2

# Reward coefficients
GOAL_REWARD = 3000
INVALID_MOVE_PENALTY = -1
PROGRESS_REWARD_FACTOR = 10
FREQUENCY_REWARD_CLIP = 1
WIND_PENALTY_VALUE = -1.0
ALIGNMENT_PENALTY_FACTOR = -2.0 / math.pi # From -1.0 * (angle_diff / np.pi) but simplified
FUEL_PENALTY_SCALE = -0.001
ETA_PENALTY_SCALE = -0.001
SPEED_BONUS_FACTOR = 0.3
BASE_STEP_PENALTY = -1
REVISIT_PENALTY_VALUE = 1.5
LOITERING_PENALTY_VALUE = 1
# Fuel consumption parameters
FUEL_SPEED_FACTOR = 0.05
FUEL_WIND_FACTOR = 0.02
FUEL_DRAG_FACTOR = 0.5

# Observation space normalization
OBS_LOW_BOUND = 0
OBS_HIGH_BOUND = 1

# CSV Logging
CSV_BUFFER_SIZE = 10000
CSV_FILE_PATH = "environment_log.csv"

# Training parameters
DEFAULT_HISTORY_LEN = 8
CALLBACK_CHECK_INTERVAL = 5000
DEFAULT_PATIENCE = 40
DEFAULT_MIN_DELTA = 2.0
MIN_FORWARD_ANGLE = 30  # degrees
MAX_FORWARD_ANGLE = 120  # degrees


# ---------------------------------------
# Parallel CNN Block with Residual
# ---------------------------------------
class ParallelCNNBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, pool_size: int = 1):
        super().__init__()
        padding = dilation * (kernel_size // 2)

        # Standard (non-dilated) conv branch
        self.standard_conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.MaxPool1d(pool_size)
        )

        # Dilated conv branch
        self.dilated_conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.MaxPool1d(pool_size)
        )

        # Optional residual projection if dimensions change
        self.use_residual = (in_channels != out_channels)
        self.residual_proj = nn.Conv1d(in_channels, out_channels, kernel_size=1) if self.use_residual else nn.Identity()

    def forward(self, x):  # x: [B, C_in, T]
        residual = self.residual_proj(x)  # Project if needed
        out = self.standard_conv(x) + self.dilated_conv(x)  # Merge both branches
        return out + residual  # Residual sum

# ---------------------------------------
# MinGRU Backbone with CNN Preprocessing
# ---------------------------------------
class MinGRUPolicyBackbone(nn.Module):
    def __init__(
        self,
        input_dim: int,
        gru_dim: int = 128,
        dropout_rate: float = 0.1,
        use_layer_norm: bool = True
    ):
        super().__init__()
        self.embedding = nn.Linear(input_dim, gru_dim)  # Embed per-frame input

        # CNN preprocessor blocks with increasing dilation
        self.cnn_block1 = ParallelCNNBlock(gru_dim, gru_dim, kernel_size=3, dilation=1)
        self.cnn_block2 = ParallelCNNBlock(gru_dim, gru_dim, kernel_size=5, dilation=2)
        # Optional third block:
        # self.cnn_block3 = ParallelCNNBlock(gru_dim, gru_dim, kernel_size=7, dilation=3)

        # MinGRU layer for long-term temporal encoding
        self.min_gru = minGRU(dim=gru_dim)

        # Optional normalization and dropout
        self.norm = nn.LayerNorm(gru_dim) if use_layer_norm else nn.Identity()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, obs_seq: torch.Tensor) -> torch.Tensor:
        """
        obs_seq: [B, T, input_dim]
        returns: [B, gru_dim] — feature vector from last timestep
        """
        # Step 1: Linear projection to embedding dim
        x = self.embedding(obs_seq)           # → [B, T, D]
        x = x.permute(0, 2, 1)                # → [B, D, T] (required by Conv1d)

        # Step 2: Temporal CNN processing (multi-scale)
        x = self.cnn_block1(x)                # → [B, D, T]
        x = self.cnn_block2(x)                # → [B, D, T]

        # Step 3: Revert to [B, T, D] for MinGRU
        x = x.permute(0, 2, 1)                # → [B, T, D]

        # Step 4: MinGRU over sequence
        x = self.min_gru(x)                   # → [B, T, D]
        x = x[:, -1, :]                       # Use last timestep: → [B, D]

        # Step 5: Final norm and dropout
        return self.dropout(self.norm(x))     # → [B, D]

# ---------------------------------------
# SB3-Compatible Features Extractor
# ---------------------------------------
class MinGRUFeaturesExtractor(BaseFeaturesExtractor):
    """
    Stable Baselines3-compatible features extractor using:
    - CNN for short-term patterns
    - MinGRU for long-term memory
    """
    def __init__(
        self,
        observation_space: spaces.Box,
        seq_len: int = 10,
        gru_dim: int = 128,
        dropout: float = 0.1,
        use_layer_norm: bool = True
    ):
        flat_dim = observation_space.shape[0]
        if flat_dim % seq_len != 0:
            raise ValueError(
                f"Expected obs dim divisible by seq_len: got {flat_dim=} and {seq_len=}"
            )

        feature_dim_per_step = flat_dim // seq_len
        super().__init__(observation_space, features_dim=gru_dim)

        self.seq_len = seq_len
        self.feature_dim_per_step = feature_dim_per_step

        self.backbone = MinGRUPolicyBackbone(
            input_dim=feature_dim_per_step,
            gru_dim=gru_dim,
            dropout_rate=dropout,
            use_layer_norm=use_layer_norm
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        obs: [B, seq_len * feature_dim_per_step] → reshaped to [B, T, F]
        """
        batch_size = obs.shape[0]
        obs_seq = obs.view(batch_size, self.seq_len, self.feature_dim_per_step)
        return self.backbone(obs_seq)

class AdamWPolicy(ActorCriticPolicy):
    def _make_optimizer(self):
        # Override the optimizer with AdamW
        self.optimizer = AdamW(
            self.parameters(),
            lr=self.lr_schedule(1),      # Required by SB3's learning rate scheduler
            weight_decay=1e-2            # ✅ This is the key change
        )

class TankerEnvironment(gym.Env):
    """
    A reinforcement learning environment for simulating a tanker's movement
    across H3 hexagonal grid cells, considering wind conditions and fuel consumption.
    """
    def __init__(self, graph: nx.Graph, wind_map: dict, start_h3: str, goal_h3: str,
                 h3_resolution: int = H3_RESOLUTION, wind_threshold: float = DEFAULT_WIND_THRESHOLD,render_mode: str = None):
        """
        Initializes the TankerEnvironment.

        Args:
            graph (nx.Graph): A NetworkX graph representing navigable H3 cells.
                              Edges should ideally have a 'weight' attribute indicating
                              visit frequency or other relevant metric.
            wind_map (dict): A dictionary where keys are H3 cell IDs and values are
                             another dictionary: {timestamp: (u_wind, v_wind)}.
                             'u_wind' is the east-west wind component, 'v_wind' is the north-south.
            start_h3 (str): The H3 index of the starting hexagon for the tanker.
            goal_h3 (str): The H3 index of the target hexagon the tanker must reach.
            h3_resolution (int): The H3 resolution for the grid. Defaults to 6.
            wind_threshold (float): The wind speed (in m/s) above which a penalty is applied.
        """
        super().__init__()

        self.graph = graph
        self.valid_hex_ids = set(self.graph.nodes)
        self.wind_map = wind_map
        self.wind_threshold = wind_threshold

        self.start_h3 = start_h3
        self.goal_h3 = goal_h3
        self.h3_resolution = h3_resolution
        self.render_mode = render_mode

        # Validate Start/Goal Nodes Are Reachable
        if start_h3 not in self.graph or goal_h3 not in self.graph:
            raise ValueError("Start or goal H3 cell is not present in the navigable water graph.")
        
        try:
            self.max_distance = nx.shortest_path_length(self.graph, start_h3, goal_h3)
        except nx.NetworkXNoPath:
            raise ValueError(f"No path exists from {start_h3} to {goal_h3} in water-only graph.")

        if self.max_distance > 100:
            self.horizon = int(self.max_distance * DEFAULT_HORIZON_MULTIPLIER)
        elif self.max_distance > 50:
            self.horizon = int(self.max_distance * 1.75)
        else:
            self.horizon = int(self.max_distance * 1.5)
        
        self.current_h3: str = start_h3
        self.prev_h3: str | None = None
        self.step_count: int = 0
        self.episode_reward: float = 0.0
        self.current_time: datetime = datetime(2018, 4, 1, 0, 0)
        self.trajectory: list[str] = []

        self._set_latlon_bounds()
        ring_radius = 2
        num_neighbors = len(h3.grid_disk(self.start_h3, ring_radius)) - 1  # exclude self
        self.speed_range = (MIN_SPEED_KNOTS, MAX_SPEED_KNOTS)

        sample_obs = self._get_observation()
        print("DEBUG: obs shape =", sample_obs.shape)


        # self.observation_space = spaces.Box(low=OBS_LOW_BOUND, high=OBS_HIGH_BOUND, shape=(4,), dtype=np.float32)

        self.k = 5 # Maximum number of neighbors to consider for action space
        obs_dim = len(sample_obs)

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

        self.action_space = spaces.MultiDiscrete([self.k, NUM_SPEED_LEVELS])
        self._distance_cache = {}

        # CSV logging attributes (currently commented out in step, can be reactivated)
        self.csv_buffer: list[dict] = []
        self.csv_buffer_size = CSV_BUFFER_SIZE
        self.csv_file_path = CSV_FILE_PATH
        self.csv_header_written = False
        self.reset()

    def _shortest_path_length(self, src: str, dst: str) -> int:
        key = (src, dst)
        if key not in self._distance_cache:
            if src not in self.graph or dst not in self.graph:
                raise ValueError(f"[Path Error] {src} or {dst} not in graph.")
            try:
                self._distance_cache[key] = nx.shortest_path_length(self.graph, src, dst)
            except nx.NetworkXNoPath:
                raise RuntimeError(f"[Path Error] No path from {src} to {dst}")
        return self._distance_cache[key]

    
    def _set_latlon_bounds(self) -> None:
        """
        Calculates and sets the min/max latitude and longitude of all H3 cells
        in the graph. These bounds are used for normalizing observations.
        """
        lats, lons = zip(*[h3.cell_to_latlng(h) for h in self.graph.nodes])
        self.min_lat, self.max_lat = min(lats), max(lats)
        self.min_lon, self.max_lon = min(lons), max(lons)

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        """
        Resets the environment to its initial state for a new episode.

        Args:
            seed (int, optional): Seed for the random number generator.
            options (dict, optional): Additional options for resetting the environment.

        Returns:
            tuple: A tuple containing the initial observation and an info dictionary.
        """
        super().reset(seed=seed)
        self.current_h3 = self.start_h3
        self.prev_h3 = None
        self.step_count = 0
        self.episode_reward = 0.0

        # Random day in August 2018 (from 1 to 29) for time initialization
        random_day = np.random.randint(1, 30)
        random_hour = np.random.randint(0, 24)
        self.current_time = datetime(2018, 8, random_day, random_hour, 0)

        self.trajectory = []

        # ✅ Track visited H3 cells and recent history to prevent loops
        self.visited_set = set()
        self.visited_set.add(self.current_h3)

        self.loiter_memory = [self.current_h3]
        self.loiter_window = 4  # Tuneable: how far back to avoid in recent loop detection

        # ✅ Optionally track edges to avoid frequency farming
        self.visited_edges = set()

        return self._get_observation(), {}


    def _get_valid_neighbors(self) -> list[str]:
        original_neighbors = list(self.graph.neighbors(self.current_h3))

        # If no previous move, don't filter by direction
        if not self.prev_h3:
            return [n for n in original_neighbors if n not in self.loiter_memory] or original_neighbors

        # Compute vessel movement direction (in radians)
        lat1, lon1 = h3.cell_to_latlng(self.prev_h3)
        lat2, lon2 = h3.cell_to_latlng(self.current_h3)
        move_dir = np.arctan2(lat2 - lat1, lon2 - lon1)

        filtered_neighbors = []
        for n in original_neighbors:
            lat_n, lon_n = h3.cell_to_latlng(n)
            neighbor_dir = np.arctan2(lat_n - lat2, lon_n - lon2)

            angle_diff = np.abs(np.arctan2(np.sin(neighbor_dir - move_dir), np.cos(neighbor_dir - move_dir)))

            if np.radians(MIN_FORWARD_ANGLE) <= angle_diff <= np.radians(MAX_FORWARD_ANGLE):
                filtered_neighbors.append(n)

        pruned_neighbors = [n for n in filtered_neighbors if n not in self.loiter_memory]

        # Fallback if filtering removed everything
        return pruned_neighbors or original_neighbors


    def _calculate_rewards(self, distance_before: int, distance_after: int,
                       speed: float, wind_speed: float, move_direction: float,
                       wind_direction: float, override_reward: bool) -> float:
        if override_reward:
            return INVALID_MOVE_PENALTY

        # 1. Distance progress
        progress_reward = PROGRESS_REWARD_FACTOR * (distance_before - distance_after)

        # 2. Visit frequency reward
        edge_id = (self.prev_h3, self.current_h3)
        if edge_id in self.visited_edges:
            frequency_reward = 0.0
        else:
            edge_weight = self.graph[self.prev_h3][self.current_h3].get('weight', 1)
            frequency_reward = np.clip(np.log1p(edge_weight) / 5, 0, FREQUENCY_REWARD_CLIP)
            self.visited_edges.add(edge_id)

        # 3. Wind speed penalty
        wind_penalty = WIND_PENALTY_VALUE if wind_speed > self.wind_threshold else 0.0

        # 4. Direction alignment penalty (penalize if heading is opposite of wind)
        angle_diff = np.arccos(np.clip(np.cos(move_direction - wind_direction), -1.0, 1.0))
        alignment_penalty = ALIGNMENT_PENALTY_FACTOR * angle_diff  # angle_diff in radians

        # 5. Fuel consumption cost
        fuel_consumed = self._estimate_fuel(speed, wind_speed, move_direction, wind_direction)
        fuel_penalty = FUEL_PENALTY_SCALE * fuel_consumed

        # 6. Time-to-arrival penalty
        travel_time = self._estimate_travel_time(speed)
        eta_penalty = ETA_PENALTY_SCALE * travel_time

        # 7. Loitering & revisits
        loiter_penalty = -LOITERING_PENALTY_VALUE if self.current_h3 in self.loiter_memory[:-1] else 0.0
        revisit_penalty = -REVISIT_PENALTY_VALUE if self.current_h3 in self.visited_set else 0.0

        # 8. Combine everything
        reward = (
            progress_reward +
            frequency_reward +
            wind_penalty +
            alignment_penalty +
            fuel_penalty +
            eta_penalty +
            BASE_STEP_PENALTY +
            loiter_penalty +
            revisit_penalty
        )

        return reward


    def _get_current_wind_conditions(self) -> tuple[float, float, float, float]:
        """
        Retrieves wind conditions at the current H3 cell and simulation time.

        Returns:
            wind_u (float): Zonal wind component (east-west)
            wind_v (float): Meridional wind component (north-south)
            wind_speed (float): Magnitude in m/s
            wind_direction (float): Direction in radians (from -π to π), CCW from East
        """
        h3_wind_data = self.wind_map.get(self.current_h3, {})

        if not h3_wind_data:
            return 0.0, 0.0, 0.0, 0.0  # No wind data available

        try:
            target_time = pd.Timestamp(self.current_time)
            closest_time = min(h3_wind_data.keys(), key=lambda t: abs(t - target_time))
            wind_u, wind_v = h3_wind_data[closest_time]
        except Exception as e:
            print(f"[WARN] Wind data issue at {self.current_h3}: {e}")
            return 0.0, 0.0, 0.0, 0.0

        wind_speed = np.sqrt(wind_u**2 + wind_v**2)
        wind_direction = np.arctan2(wind_v, wind_u)  # Returns [-π, π]

        return wind_u, wind_v, wind_speed, wind_direction


    def step(self, action: tuple[int, int]) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Takes one step in the environment given an action.
        """
        info = {}
        override_reward = False
        neighbor_idx, speed_level = action
        speed = np.linspace(self.speed_range[0], self.speed_range[1], NUM_SPEED_LEVELS)[speed_level]

        neighbors = self._get_valid_neighbors()
        if not neighbors:
            return self._get_observation(), INVALID_MOVE_PENALTY, False, True, {
                "reason": "no valid neighbors"
            }

        # ✅ Strict check — prevent accessing invalid index
        if neighbor_idx >= len(neighbors):
            return self._get_observation(), INVALID_MOVE_PENALTY, False, True, {
                "reason": f"invalid neighbor index {neighbor_idx}, only {len(neighbors)} valid"
            }

        selected_neighbor = neighbors[neighbor_idx]

        if selected_neighbor == self.current_h3:
            override_reward = True

        try:
            distance_before = self._shortest_path_length(self.current_h3, self.goal_h3)

            self.prev_h3 = self.current_h3
            self.current_h3 = selected_neighbor
            self.step_count += 1

            distance_after = self._shortest_path_length(self.current_h3, self.goal_h3)

            # 🧠 Update visited sets
            self.visited_set.add(self.current_h3)
            self.loiter_memory.append(self.current_h3)
            if len(self.loiter_memory) > self.loiter_window:
                self.loiter_memory.pop(0)

        except nx.NetworkXNoPath:
            return self._get_observation(), INVALID_MOVE_PENALTY, False, True, {
                "reason": "no ocean path to goal",
                "current_h3": self.current_h3,
                "selected_neighbor": selected_neighbor,
                "goal_h3": self.goal_h3,
                "step_count": self.step_count
            }

        # 🌬 Wind
        wind_u, wind_v, wind_speed, wind_direction = self._get_current_wind_conditions()
        
        # ⛵ Movement direction
        if self.prev_h3:
            lat1, lon1 = h3.cell_to_latlng(self.prev_h3)
            lat2, lon2 = h3.cell_to_latlng(self.current_h3)
            move_direction = np.arctan2(lat2 - lat1, lon2 - lon1)
        else:
            move_direction = 0.0

        # ⛽ Fuel + ⏱ ETA + ⛵ Alignment
        fuel_used = self._estimate_fuel(speed, wind_speed, move_direction, wind_direction)
        travel_time = self._estimate_travel_time(speed)
        alignment_angle = np.arccos(np.clip(np.cos(move_direction - wind_direction), -1.0, 1.0))
        eta_penalty = ETA_PENALTY_SCALE * travel_time

        # 🎯 Reward
        reward = self._calculate_rewards(
            distance_before, distance_after,
            speed, wind_speed, move_direction,
            wind_direction, override_reward
        )

        if self.current_h3 == self.goal_h3:
            if self.render_mode == "debug":
                print(f"[DEBUG] Goal reached at step {self.step_count}")
            reward += GOAL_REWARD
            terminated = True
        else:
            terminated = False

        truncated = self.step_count >= self.horizon
        done = terminated or truncated
        self.episode_reward += reward
        self.trajectory.append(self.current_h3)

        if done:
            info["episode"] = {"r": self.episode_reward, "l": self.step_count}
            info["distance_before"] = distance_before
            info["distance_after"] = distance_after

        info.update({
            "terminated": terminated,
            "truncated": truncated,
            "step_count": self.step_count,
            "current_h3": self.current_h3,
            "prev_h3": self.prev_h3,
            "distance_to_goal": distance_after,
            "progress": PROGRESS_REWARD_FACTOR * (distance_before - distance_after),
            "override_reward": override_reward,
            "wind_penalty": WIND_PENALTY_VALUE if wind_speed > self.wind_threshold else 0.0,
            "alignment_penalty": ALIGNMENT_PENALTY_FACTOR * alignment_angle,
            "fuel_penalty": FUEL_PENALTY_SCALE * fuel_used,
            "fuel_consumed": fuel_used,
            "eta_penalty": eta_penalty,
            "speed": speed,
            "wind_direction": wind_direction,
            "move_direction": move_direction,
            "angle_diff": alignment_angle,
            "current_time": self.current_time,
            "raw_reward": reward,
            "reward_per_step": reward
        })

        self.current_time += timedelta(seconds=travel_time)
        return self._get_observation(speed), reward, terminated, truncated, info


    def _estimate_fuel(self, speed: float, wind_speed: float,
                       move_direction: float, wind_direction: float) -> float:
        """
        Estimates the fuel consumed during a step based on vessel speed, wind speed,
        and the alignment between vessel movement and wind direction.

        Args:
            speed (float): Vessel speed in knots.
            wind_speed (float): Wind speed magnitude in m/s.
            move_direction (float): Ship's heading in radians.
            wind_direction (float): Wind direction in radians.

        Returns:
            float: Estimated fuel consumed.
        """
        # Calculate a factor based on the alignment of movement and wind.
        # When aligned, cos(0) = 1, angle_factor = 2. When opposite, cos(pi) = -1, angle_factor = 0.
        angle_factor = 1 + np.cos(move_direction - wind_direction)
        # Adjust drag: worse (higher) when sailing against the wind (angle_factor is low)
        adjusted_drag = 1 + FUEL_DRAG_FACTOR * (1 - angle_factor)
        # Fuel consumption formula: (speed^3 component adjusted by drag) + (wind_speed component)
        return FUEL_SPEED_FACTOR * (speed ** 3) * adjusted_drag + FUEL_WIND_FACTOR * wind_speed

    def _estimate_travel_time(self, speed: float) -> float:
        """
        Estimates the time (in seconds) it takes to travel between two adjacent
        H3 cells based on the given speed and the average edge length of an H3 cell.

        Args:
            speed (float): Vessel speed in knots.

        Returns:
            float: Estimated travel time in seconds.
        """
        meters_per_h3_grid_unit = h3.average_hexagon_edge_length(self.h3_resolution, unit='m')

        grid_distance = 1 if self.prev_h3 is None else h3.grid_distance(self.prev_h3, self.current_h3)

        distance_m = grid_distance * meters_per_h3_grid_unit

        speed_mps = speed * KNOTS_TO_MPS
        return distance_m / max(speed_mps, 1e-6) # Avoid division by zero

    def _get_observation(self, speed: float = MIN_SPEED_KNOTS) -> np.ndarray:
        lat, lon = h3.cell_to_latlng(self.current_h3)
        norm_lat = np.clip((lat - self.min_lat) / max(self.max_lat - self.min_lat, 1e-6), 0, 1)
        norm_lon = np.clip((lon - self.min_lon) / max(self.max_lon - self.min_lon, 1e-6), 0, 1)

        log_min = np.log10(self.speed_range[0] + 1e-5)
        log_max = np.log10(self.speed_range[1] + 1e-5)
        norm_speed = np.clip((np.log10(speed + 1e-5) - log_min) / (log_max - log_min), 0, 1)

        _, _, _, wind_direction = self._get_current_wind_conditions()
        norm_wind_angle = np.clip((wind_direction + np.pi) / (2 * np.pi), 0, 1)

        raw_wind_field = self._get_local_wind_field(self.current_h3, radius=2)
        normalized_wind_field = []
        MAX_WIND_SPEED = 30.0

        for i in range(0, len(raw_wind_field), 4):
            delta_lat = raw_wind_field[i]
            delta_lon = raw_wind_field[i + 1]
            wind_speed = raw_wind_field[i + 2]
            norm_dir = raw_wind_field[i + 3]

            norm_delta_lat = np.clip(delta_lat, -1, 1)
            norm_delta_lon = np.clip(delta_lon, -1, 1)
            norm_speed_local = np.clip(wind_speed / MAX_WIND_SPEED, 0, 1)

            wind_dir = norm_dir * 2 * np.pi - np.pi
            sin_dir = np.sin(wind_dir)
            cos_dir = np.cos(wind_dir)

            normalized_wind_field.extend([
                norm_delta_lat,
                norm_delta_lon,
                norm_speed_local,
                sin_dir,
                cos_dir
            ])

        obs = [norm_lat, norm_lon, norm_speed, norm_wind_angle] + normalized_wind_field
        obs_array = np.asarray(obs, dtype=np.float32)
        obs_array = np.nan_to_num(obs_array, nan=0.0, posinf=OBS_HIGH_BOUND, neginf=OBS_LOW_BOUND)
        assert np.all(np.isfinite(obs_array)), f"Non-finite obs: {obs_array}"

        return obs_array


    def render(self, mode: str = 'human') -> np.ndarray | None:
        import io
        import contextily as ctx
        import matplotlib.pyplot as plt
        from shapely.geometry import Polygon
        import geopandas as gpd

        # Cache static ocean plot
        if not hasattr(self, "_cached_ocean_gdf"):
            cell_polys = [
                {"h3": h, "geometry": Polygon(h3.cell_to_boundary(h))}
                for h in self.valid_hex_ids
            ]
            self._cached_ocean_gdf = gpd.GeoDataFrame(cell_polys, crs="EPSG:4326").to_crs(epsg=3857)

        traj_polys = [
            {"h3": h, "geometry": Polygon(h3.cell_to_boundary(h))}
            for h in self.trajectory
        ]
        traj_gdf = gpd.GeoDataFrame(traj_polys, crs="EPSG:4326").to_crs(epsg=3857)

        current_poly = Polygon(h3.cell_to_boundary(self.current_h3))
        current_gdf = gpd.GeoDataFrame([{"geometry": current_poly}], crs="EPSG:4326").to_crs(epsg=3857)

        fig, ax = plt.subplots(figsize=(6, 6))
        self._cached_ocean_gdf.plot(ax=ax, facecolor="lightblue", edgecolor="gray", linewidth=0.2)
        traj_gdf.plot(ax=ax, facecolor="orange", edgecolor="black", linewidth=0.5)
        current_gdf.plot(ax=ax, facecolor="red", edgecolor="black")
        ctx.add_basemap(ax, source=ctx.providers.Stamen.TonerLite)
        ax.set_axis_off()
        plt.tight_layout()

        if mode == 'rgb_array':
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100)
            plt.close(fig)
            buf.seek(0)
            img = plt.imread(buf)
            return img  # shape (H, W, 4) — RGBA
        elif mode == 'human':
            plt.title(f"Step {self.step_count} | Time: {self.current_time.strftime('%Y-%m-%d %H:%M')}")
            plt.show()
            plt.close(fig)
        else:
            super().render(mode=mode)

    def _get_local_wind_field(self, center_h3: str, radius: int = 2) -> list[float]:
        lat0, lon0 = h3.cell_to_latlng(center_h3)
        field = []

        ring_cells = set(h3.grid_disk(center_h3, radius))

        ring_cells.discard(center_h3)

        for h3_cell in sorted(ring_cells):  # sorted for consistency
            lat, lon = h3.cell_to_latlng(h3_cell)
            delta_lat = lat - lat0
            delta_lon = lon - lon0

            wind_data = self.wind_map.get(h3_cell, {})
            if wind_data:
                target_time = pd.Timestamp(self.current_time)
                closest_time = min(wind_data.keys(), key=lambda t: abs(t - target_time))
                wind_u, wind_v = wind_data[closest_time]
                wind_speed = np.sqrt(wind_u**2 + wind_v**2)
                wind_dir = np.arctan2(wind_v, wind_u)
            else:
                wind_speed = 0.0
                wind_dir = 0.0

            norm_wind_dir = (wind_dir + np.pi) / (2 * np.pi)
            field.extend([delta_lat, delta_lon, wind_speed, norm_wind_dir])

        return field

    def buffer_selected_info(self, info: dict) -> None:
        """
        Buffers selected information from the info dictionary to be written to a CSV file.
        """
        selected_info = {
            "distance_to_goal": info["distance_to_goal"],
            "progress": info["progress"],
            "override_reward": info["override_reward"],
            "wind_penalty": info["wind_penalty"],
            "alignment_penalty": info["alignment_penalty"],
            "fuel_penalty": info["fuel_penalty"],
            "speed": info["speed"],
            "fuel_consumed": info["fuel_consumed"],
            "raw_reward": info["raw_reward"],
            "current_time": info["current_time"],
            "wind_direction": info["wind_direction"],
            "move_direction": info["move_direction"],
            "angle_diff": info["angle_diff"],
            "eta_penalty": info["eta_penalty"],
        }

        self.csv_buffer.append(selected_info)

        if len(self.csv_buffer) >= self.csv_buffer_size:
            self.flush_buffer_to_csv()

    def flush_buffer_to_csv(self) -> None:
        """
        Writes the buffered step information to a CSV file.
        """
        if not self.csv_buffer:
            return

        write_header = not self.csv_header_written
        with open(self.csv_file_path, mode='a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.csv_buffer[0].keys())
            if write_header:
                writer.writeheader()
                self.csv_header_written = True
            writer.writerows(self.csv_buffer)
        self.csv_buffer.clear()
    
class TankerEnvWithHistory(gym.Wrapper):
    """
    A Gym wrapper to append a history of observations to the current observation.
    This is useful for models like MinGRU that process sequences.
    """
    def __init__(self, env: gym.Env, history_len: int = DEFAULT_HISTORY_LEN):
        super().__init__(env)
        self.history_len = history_len
        self.obs_buffer: list[np.ndarray] = []

        obs_space = env.observation_space
        # The observation space for the wrapped environment is flattened
        # to be compatible with `MinGRUFeaturesExtractor`.
        self.observation_space = spaces.Box(
            low=np.repeat(obs_space.low, history_len),
            high=np.repeat(obs_space.high, history_len),
            shape=(obs_space.shape[0] * history_len,), # Flattened shape
            dtype=np.float32
        )

    def reset(self, **kwargs) -> tuple[np.ndarray, dict]:
        """
        Resets the environment and initializes the observation buffer with
        the first observation repeated `history_len` times.
        """
        obs, info = self.env.reset(**kwargs)
        self.obs_buffer = [obs] * self.history_len
        return np.concatenate(self.obs_buffer, axis=0), info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Takes a step in the environment, appends the new observation to the buffer,
        and returns the concatenated history as the observation.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.obs_buffer.append(obs)
        self.obs_buffer = self.obs_buffer[-self.history_len:] # Keep only the latest `history_len` observations
        stacked_obs = np.concatenate(self.obs_buffer, axis=0)
        return stacked_obs, reward, terminated, truncated, info

class EarlyStoppingCallback(BaseCallback):
    def __init__(self, log_path: str, patience: int = 10, min_delta: float = 1.0,
                 check_freq: int = 10000, verbose: int = 1):
        super().__init__(verbose)
        self.log_path = log_path
        self.patience = patience
        self.min_delta = min_delta
        self.check_freq = check_freq
        self.counter = 0
        self.best_mean_reward = -np.inf
        self.start_time = time.time()
        self.pbar = None
        self.log_file = os.path.join(log_path, "earlystop.log")

    def _on_step(self) -> bool:
        # Create tqdm only if in TTY
        if self.num_timesteps >= self.check_freq and self.pbar is None:
            self.pbar = tqdm(
                total=None,
                desc="Training Progress",
                position=0,
                dynamic_ncols=True,
                ascii=not sys.stdout.isatty(),
                disable=not sys.stdout.isatty()
            )

        if self.pbar is not None:
            step_increment = getattr(self.model, "n_steps", 2048)
            self.pbar.update(step_increment)

        if self.num_timesteps % self.check_freq == 0:
            try:
                monitor_file = os.path.join(self.log_path, "evaluations.npz")
                if os.path.exists(monitor_file):
                    data = np.load(monitor_file)
                    rewards = data["results"]
                    if rewards.shape[0] > 0:
                        latest_mean = np.mean(rewards[-1])
                        elapsed = time.time() - self.start_time

                        # Write to both tqdm-safe output and a permanent log file
                        msg = (f"[EarlyStopping] Step {self.num_timesteps} | "
                               f"Mean Eval Reward = {latest_mean:.2f} | "
                               f"Elapsed Time = {elapsed:.1f}s")
                        tqdm.write(msg)
                        with open(self.log_file, "a") as f:
                            f.write(f"{self.num_timesteps},{latest_mean:.2f},{elapsed:.1f}\n")

                        if latest_mean > self.best_mean_reward + self.min_delta:
                            tqdm.write(f"✅ New best model! Mean reward improved from "
                                       f"{self.best_mean_reward:.2f} to {latest_mean:.2f}")
                            self.best_mean_reward = latest_mean
                            self.counter = 0
                        else:
                            self.counter += 1
                            tqdm.write(f"[EarlyStopping] No improvement for {self.counter} evaluations")

                        if self.counter >= self.patience:
                            tqdm.write("🛑 Early stopping triggered.")
                            if self.pbar:
                                self.pbar.close()
                            return False
            except Exception as e:
                tqdm.write(f"[EarlyStopping] Failed to read eval log: {e}")

        return True

    def _on_training_end(self) -> None:
        if self.pbar:
            self.pbar.close()

def load_full_wind_map(csv_path: str) -> dict[str, dict[datetime, tuple[float, float]]]:
    """
    Loads wind data from a CSV file into a nested dictionary structure.

    Args:
        csv_path (str): The path to the CSV file containing wind data.

    Returns:
        dict: A dictionary where keys are H3 cell IDs and values are
              dictionaries mapping timestamps to (u_wind, v_wind) tuples.
    """
    wind_map: defaultdict[str, dict[datetime, tuple[float, float]]] = defaultdict(dict)

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            h3_cell = row["h3_cell"]
            timestamp = datetime.fromisoformat(row["timestamp"])
            u = float(row["u"])
            v = float(row["v"])
            wind_map[h3_cell][timestamp] = (u, v)

    return dict(wind_map)

def smooth(data: list[float], weight: float = 0.9) -> list[float]:
    """
    Applies an Exponential Moving Average (EMA) to smooth a list of numerical data.

    Args:
        data (list[float]): The list of numerical data to smooth.
        weight (float): The smoothing factor (between 0 and 1). Higher weight means
                        more smoothing (less responsive to new data).

    Returns:
        list[float]: The smoothed data.
    """
    smoothed = []
    if not data:
        return smoothed
    last = data[0]
    for point in data:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def make_env():
    def _init():
        base_env = TankerEnvironment(
            start_h3=start_h3,
            goal_h3=goal_h3,
            graph=G_visits,
            wind_map=full_wind_map,
            h3_resolution=H3_RESOLUTION,
            wind_threshold=22,
            render_mode="human"
        )
        env = TankerEnvWithHistory(base_env, history_len=DEFAULT_HISTORY_LEN)
        return Monitor(env)  # 👈 wrap with Monitor
    return _init

class StepRewardLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        for i, info in enumerate(self.locals["infos"]):
            if "reward_per_step" in info:
                self.logger.record("reward/step_reward", info["reward_per_step"])
        return True

if __name__ == "__main__":
    mp.set_start_method("fork", force=True)
    start_h3 = "862b256dfffffff"
    goal_h3 = "862b160d7ffffff"
    wind_map_path = "august_2018_60min_windmap_v2.csv"
    graph_path = "GULF_VISITS_CARGO_TANKER_AUGUST_2018.gexf"

    print(f"Loading wind map from {wind_map_path}...")
    full_wind_map = load_full_wind_map(wind_map_path)
    print(f"Loading graph from {graph_path}...")
    G_visits = nx.read_gexf(graph_path).to_undirected()
    print("Data loading complete.")

    base_env = TankerEnvironment(
        start_h3=start_h3,
        goal_h3=goal_h3,
        graph=G_visits,
        wind_map=full_wind_map,
        h3_resolution=H3_RESOLUTION,
        wind_threshold=22 # Example wind threshold
    )

    # 1. Wrap the base environment to include observation history
    env_history_len = DEFAULT_HISTORY_LEN
    env = TankerEnvWithHistory(base_env, history_len=env_history_len)
    envs = 3
    vec_env = SubprocVecEnv([make_env() for _ in range(envs)])
    # vec_env = DummyVecEnv([make_env() for _ in range(envs)])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=5.0, clip_reward=10.0)
    
    # 2. Define policy architecture and features extractor
    policy_kwargs = dict(
    features_extractor_class=MinGRUFeaturesExtractor,
    features_extractor_kwargs=dict(seq_len=16),  # 👈 pick 16 or 32
    net_arch=dict(pi=[64, 64], vf=[64, 64]),
    activation_fn=nn.GELU
)

    # Define a log directory for TensorBoard
    log_dir = f"new_runs/Test_2_Conv_MinGRU_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ent_coef_schedule = get_schedule_fn(get_linear_fn(start=0.02, end=0.001, end_fraction=1.0))
    # Start at 3e-4 and end at 3e-5 by the end of training
    learning_rate_schedule = get_linear_fn(start=3e-4, end=3e-5, end_fraction=1.0)

    # 3. Instantiate PPO model
    model = PPO(
    policy=AdamWPolicy,         
    env=vec_env,
    policy_kwargs=policy_kwargs,
    learning_rate=learning_rate_schedule,
    n_steps=2048,
    batch_size=256,
    n_epochs=10,
    ent_coef=0.05,
    vf_coef=0.5,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.1,
    max_grad_norm=0.5,
    tensorboard_log=log_dir,
    verbose=1,
    device="cuda"
)

    # 4. Print the model architecture for inspection
    print("\n--- 🔧 Model Architecture ---")
    print(model.policy)
    for name, param in model.policy.named_parameters():
        print(f"{name:<40} {list(param.shape)}")
    print("----------------------------\n")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    eval_callback = EvalCallback(
    eval_env=vec_env,  # Wrap with Monitor
    best_model_save_path=f"./Test_2_{timestamp}",
    log_path="./eval_logs",  # important!
    eval_freq=10000,
    deterministic=True,
    render=False
    )
    
    # Set up Early Stopping Callback
    early_stop = EarlyStoppingCallback(
    log_path="./eval_logs",
    patience=200,
    min_delta=2.0,
    check_freq=10000
    )

    step_logger = StepRewardLoggerCallback()

    callback = CallbackList([eval_callback, early_stop,step_logger])

    # Train the model
    print(f"Starting training for {12000000} timesteps...")
    model.learn(total_timesteps=12000000, callback=callback)
    print("Training finished.") 
