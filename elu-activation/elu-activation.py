import math
def elu(x, alpha):
    """
    Apply ELU activation to each element.
    """
    result = []
    for x_each in x:
          if x_each > 0:
              result.append(x_each)
          else:
              result.append(alpha * (math.exp(x_each) - 1))
    return result