import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """
    PE = np.zeros((seq_len, d_model))                                      
    pos = np.arange(seq_len).reshape(-1, 1)
    n_sin = (d_model + 1) // 2   # ceil，奇数时比 cos 多1
    n_cos = d_model // 2          # floor

    i_sin = np.arange(n_sin).reshape(1, -1)
    i_cos = np.arange(n_cos).reshape(1, -1)

    PE[:, 0::2] = np.sin(pos / np.exp(2 * i_sin / d_model * np.log(base)))
    PE[:, 1::2] = np.cos(pos / np.exp(2 * i_cos / d_model * np.log(base)))

    return PE