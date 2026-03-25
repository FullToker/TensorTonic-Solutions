import numpy as np

def dropout(x, p=0.5, rng=None):
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern).
    """
    # Write code here
    x = np.asarray(x, dtype= float)
    if rng is None:
        random_vals = np.random.random(x.shape)
    else:
        random_vals = rng.random(x.shape)
    mask = (random_vals < (1 - p)).astype(float) / (1-p)
    output = x * mask
    return output, mask
