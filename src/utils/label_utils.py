import ast
import numpy as np
import pandas as pd

from src.configs.config import *


# ============================================================
# CREATE LABEL MAP
# ============================================================

def create_label_map(train_df: pd.DataFrame):

    labels = sorted(train_df["primary_label"].unique())

    label_to_idx = {
        label: idx
        for idx, label in enumerate(labels)
    }

    idx_to_label = {
        idx: label
        for label, idx in label_to_idx.items()
    }

    return label_to_idx, idx_to_label


# ============================================================
# MULTI HOT ENCODING
# ============================================================

def create_target(
    primary_label,
    secondary_labels,
    label_to_idx
):

    target = np.zeros(
        len(label_to_idx),
        dtype=np.float32
    )

    # primary label
    target[label_to_idx[primary_label]] = 1.0

    # secondary labels
    try:

        secondary_labels = ast.literal_eval(
            secondary_labels
        )

        for label in secondary_labels:

            if label in label_to_idx:

                target[
                    label_to_idx[label]
                ] = SECONDARY_LABEL_WEIGHT

    except:
        pass

    return target