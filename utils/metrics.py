import numpy as np

def metrics_in_range(gt, pred, interval):
    # gt = array([ 11, 106, 236, 296, 367, 604])
    # pred = [122, 239, 338, 484, 525, 589, 709, 787, 915]
    if type(gt) != list:
        gt = list(gt)
    if type(pred) != list:
        pred = list(pred)
    total_pred = len(pred)
    total_gt = len(gt)
    gt = gt.copy()
    in_interval_pred = 0
    # for each predicted value
    for p in pred:
        # find a gt value
        for g in gt:
            # that's within the inverval
            if np.abs(p - g) <= interval:
                in_interval_pred += 1
                gt.remove(g)
                break
    precision = in_interval_pred / total_pred if total_pred != 0.0 else 0.0
    recall = min(in_interval_pred / total_gt, 1.0)
    f1 = 2*(recall*precision)/(recall+precision) if (recall != 0.0 and precision != 0.0) else 0.0
    return {"precision": precision, "recall": recall, "F1": f1}
