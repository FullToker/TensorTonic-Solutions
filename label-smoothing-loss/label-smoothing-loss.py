def label_smoothing_loss(predictions, target, epsilon):
    import math
    K = len(predictions)
    loss = 0.0
    for i in range(K):
          q_i = (1 - epsilon) + epsilon / K if i == target else epsilon / K
          loss += q_i * math.log(predictions[i])
    return -loss