import math

def binary_focal_loss(predictions: list, targets: list, alpha: float, gamma: float) -> float:
    """
    Returns the mean binary focal loss as a float.
    """
    # Write code here
    eps = 1e-8  # 防止log(0)导致数值问题
    n = len(predictions)
    total_loss = 0.0

    for p, y in zip(predictions, targets):
        # 1. 计算p_t
        p_t = p if y == 1 else (1 - p)
        p_t = min(max(p_t, eps), 1 - eps)  # 裁剪,避免log(0)

        # 2. 计算这个样本的focal loss: FL = -alpha * (1-p_t)^gamma * ln(p_t)
        loss = -alpha * ((1 - p_t) ** gamma) * math.log(p_t)

        total_loss += loss

    return total_loss / n