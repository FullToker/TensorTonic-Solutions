import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    # Your code here
    if not seqs:
        return np.zeros((0, max_len or 0), dtype=int) 
    if max_len is None:
        max_len = max(len(seq) for seq in seqs)
    result = np.full((len(seqs), max_len), fill_value=pad_value)
    for i, seq in enumerate(seqs):
        le = min(len(seq), max_len)
        result[i, :le] = seq[:le]
    return result