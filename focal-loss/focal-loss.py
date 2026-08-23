import numpy as np

def focal_loss(p: list, y: list, gamma: float = 2.0) -> float:
    """Return the mean binary focal loss."""
    # Write code here
    eps = 1e-8  # 防止log(0)
    n = len(p)
    total_loss = 0.0
    
    for i in range(n):
        p_i = min(max(p[i], eps), 1 - eps)  # 裁剪,避免数值问题
        y_i = y[i]
        
        # p_t: 正样本用p_i,负样本用1-p_i
        p_t = p_i if y_i == 1 else (1 - p_i)
        
        # 公式(4): FL(p_t) = -(1-p_t)^gamma * log(p_t)
        loss_i = -((1 - p_t) ** gamma) * math.log(p_t)
        
        total_loss += loss_i
    
    return total_loss / n
    