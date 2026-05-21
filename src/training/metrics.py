import numpy as np

from sklearn.metrics import roc_auc_score


def macro_auc(
    targets,
    predictions
):

    aucs = []

    num_classes = targets.shape[1]

    for i in range(num_classes):

        target_col = targets[:, i]

        pred_col = predictions[:, i]

        # Skip invalid classes
        if len(np.unique(target_col)) < 2:
            continue

        try:

            auc = roc_auc_score(
                target_col,
                pred_col
            )

            aucs.append(auc)

        except:
            continue

    if len(aucs) == 0:
        return 0.0

    return np.mean(aucs)