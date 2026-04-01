import numpy as np

def triplet_loss(anchor, positive, negative, margin=1.0):
    """
    Compute Triplet Loss for embedding ranking.
    """
    # Write code here
    anchor = np.asarray(anchor)
    positive = np.asarray(positive)
    negative = np.asarray(negative)
    # axis =-1 应对不同维度的输入
    dis_ap =  np.sum(np.square(anchor - positive), axis=-1)
    dis_an =  np.sum(np.square(anchor - negative), axis=-1)
    losses = np.maximum(0, dis_ap-dis_an+margin)
    return np.mean(losses)