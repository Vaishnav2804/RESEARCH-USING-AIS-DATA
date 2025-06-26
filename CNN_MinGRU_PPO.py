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
