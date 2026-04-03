import numpy as np

def dice_loss(p, y, eps=1e-8):
    """
    Compute Dice Loss for segmentation.
    """
    # Write code here
    p =  (np.asarray(p)).flatten()
    y = np.asarray(y).flatten()
    sum_y = np.sum(y)
    sum_p = np.sum(p)

    inter = np.sum(p * y)
    dice = ((2 * inter) + eps)/(sum_p + sum_y+eps)
    return 1-dice