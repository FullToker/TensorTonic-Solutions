import numpy as np

def gini(node):
    _, counts = np.unique(node, return_counts=True, axis=-1)
    probs = counts/counts.sum()
    return 1- np.sum(probs**2)

def gini_impurity(y_left, y_right):
    """
    Compute weighted Gini impurity for a binary split.
    """
    # Write code here
    y_left = np.asarray(y_left)
    y_right = np.asarray(y_right)
    if len(y_left)==0 and len(y_right)==0: return 0.0
    if len(y_left)==0: return gini(y_right)
    if len(y_right)==0: return gini(y_left)
    n_total = len(y_left) + len(y_right)
    return (len(y_left)/n_total)*gini(y_left)+ (len(y_right)/n_total)*gini(y_right)
    