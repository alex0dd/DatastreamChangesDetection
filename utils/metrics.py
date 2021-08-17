import numpy as np

def metrics_in_range(gt, pred, interval):
    # gt = array([ 11, 106, 236, 296, 367, 604])
    # pred = [122, 239, 338, 484, 525, 589, 709, 787, 915]
    total_pred = len(pred)
    total_gt = len(gt)
    in_interval_pred = 0
    for p in pred:
        for g in gt:
            if np.abs(p - g) <= interval:
                in_interval_pred += 1
                break
    precision = in_interval_pred / total_pred
    recall = in_interval_pred / total_gt
    f1 = 2*(recall*precision)/(recall+precision) if (recall != 0.0 and precision != 0.0) else 0.0
    return {"precision": precision, "recall": recall, "F1": f1}
