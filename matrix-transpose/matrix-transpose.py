import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    A = np.asarray(A)
    row, col = A.shape
    result = np.zeros((col, row))
    for i in range(row):
        result[: , i] = A[i, :]
    return result
    
