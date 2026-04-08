import numpy as np

def info_nce_loss(Z1, Z2, temperature=0.1):
    """
    Compute InfoNCE Loss for contrastive learning.
    """
    # Write code here
    Z1 = np.asarray(Z1, dtype=float)
    Z2 = np.asarray(Z2, dtype=float)

    S = np.dot(Z1, Z2.transpose())/temperature
    S_stable = S- np.max(S, axis=-1, keepdims=True)
    loss = -(1/Z1.shape[0]) * np.sum(np.log(np.exp(np.diag(S_stable))/np.sum(np.exp(S_stable), axis=-1)))
    return loss