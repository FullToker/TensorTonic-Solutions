import numpy as np

def classification_metrics(y_true: list[int], y_pred: list[int], average: str = "micro", pos_label: int = 1) -> dict:
    """
    Returns a dictionary containing accuracy, precision, recall, and f1 rounded to six decimals.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    accuracy = np.mean(y_true == y_pred)

    def prf_for_class(c):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        support = np.sum(y_true == c)
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        return tp, fp, fn, support, p, r, f

    if average == "binary":
        _, _, _, _, precision, recall, f1 = prf_for_class(pos_label)

    elif average == "micro":
        classes = np.unique(np.concatenate([y_true, y_pred]))
        tp_total = fp_total = fn_total = 0
        for c in classes:
            tp, fp, fn, _, _, _, _ = prf_for_class(c)
            tp_total += tp
            fp_total += fp
            fn_total += fn
        precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
        recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    elif average == "macro":
        classes = np.unique(np.concatenate([y_true, y_pred]))
        ps, rs, fs = [], [], []
        for c in classes:
            _, _, _, _, p, r, f = prf_for_class(c)
            ps.append(p)
            rs.append(r)
            fs.append(f)
        precision = np.mean(ps)
        recall = np.mean(rs)
        f1 = np.mean(fs)

    elif average == "weighted":
        classes = np.unique(np.concatenate([y_true, y_pred]))
        total = len(y_true)
        precision = recall = f1 = 0.0
        for c in classes:
            _, _, _, support, p, r, f = prf_for_class(c)
            weight = support / total
            precision += weight * p
            recall += weight * r
            f1 += weight * f

    else:
        raise ValueError(f"Unsupported average type: {average}")

    return {
        "accuracy": round(float(accuracy), 6),
        "precision": round(float(precision), 6),
        "recall": round(float(recall), 6),
        "f1": round(float(f1), 6),
    }