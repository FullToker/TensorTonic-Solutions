import numpy as np

def minmax_scale(X: list, axis: int = 0, eps: float = 1e-12) -> np.ndarray:
    """
    Returns a floating-point NumPy array matching the shape of X.
    """
    # Write code here
    x = np.asarray(X, dtype = float)
    x_min = np.min(x, axis= axis, keepdims =True)
    x_max = np.max(x, axis= axis, keepdims =True)
    return (x - x_min)/(x_max - x_min +eps)