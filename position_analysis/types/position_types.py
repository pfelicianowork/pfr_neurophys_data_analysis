from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


@dataclass
class PositionSeries1D:
    time: np.ndarray
    value: np.ndarray  # linear position or velocity (1D)


@dataclass
class PositionSeries2D:
    time: np.ndarray
    xy: np.ndarray  # shape (N,2)


@dataclass
class LinearizationResult:
    linear_pos: np.ndarray
    edge_ids: np.ndarray
    bounds: Optional[Tuple[float, float]] = None
