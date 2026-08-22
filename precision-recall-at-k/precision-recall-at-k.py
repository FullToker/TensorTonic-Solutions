def precision_recall_at_k(recommended: list, relevant: list, k: int) -> list[float]:
    """
    Return precision at k and recall at k.
    """
    top_k = recommended[:k]              # 取前k个推荐
    relevant_set = set(relevant)          # 转成集合,方便快速查找
    
    hits = len(set(top_k) & relevant_set)  # 计算交集大小:前k个里命中了几个相关物品
    
    precision = hits / k
    recall = hits / len(relevant_set) if len(relevant_set) > 0 else 0
    
    return [precision, recall]