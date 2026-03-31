import numpy as np

def normalize_3d(v):
    """
    Normalize 3D vector(s) to unit length.
    """
    # Your code here
    v = np.asarray(v, dtype=float)
    v_norm = np.linalg.norm(v, axis=-1, keepdims=True)  # shape (..., 1)
    safe_norm = np.where(v_norm > 1e-10, v_norm, 1.0)
    return v / safe_norm
