import math
import numpy as np

def ndcg(relevance_scores, k):
    """
    Compute NDCG@k.
    """
    # Write code here
    relevance_scores = np.asarray(relevance_scores, dtype=float)
    k = min(k, len(relevance_scores))

    ranks = np.arange(1, k+1)
    discounts = np.log2(ranks+1)
    gains = (2**relevance_scores[:k]-1)/discounts
    dcg = gains.sum()

    ideal = np.sort(relevance_scores)[::-1]
    ideal_gains = (2**ideal[:k]-1)/discounts

    idcg = ideal_gains.sum()

    return 0.0 if idcg==0 else dcg/idcg