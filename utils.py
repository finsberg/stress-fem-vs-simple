import numpy as np
from typing import Iterable


def laplace(
    pressure: float | Iterable,
    radius: float | Iterable,
    width: float | Iterable,
    factor=2.0,
) -> float | np.ndarray:
    return np.array(pressure) * np.array(radius) / (factor * np.array(width))
