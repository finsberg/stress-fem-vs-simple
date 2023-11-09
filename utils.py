import numpy as np


def laplace(
    pressure: float | np.ndarray,
    radius: float | np.ndarray,
    width: float | np.ndarray,
    factor=2.0,
) -> float | np.ndarray:
    return pressure * radius / (factor * width)
