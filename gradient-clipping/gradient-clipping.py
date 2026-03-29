import numpy as np

def clip_gradients(g, max_norm):
    """
    Clip gradients using global norm clipping.
    """
    # Write code here
    g= np.asarray(g)
    g_norm = np.sqrt(np.sum(g**2, keepdims=False))
    if g_norm <= max_norm or max_norm<=0:
        return g
    else:
        return g * (max_norm / g_norm)