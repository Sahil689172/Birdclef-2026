import os

import numpy as np
import pandas as pd
import torch

from tqdm import tqdm

from sklearn.metrics import f1_score

from torch.utils.data import DataLoader

from src.configs.config import *

from src.datasets.soundscape_dataset import SoundscapeDataset

from src.models.efficientnet_model import BirdCLEFModel

from src.utils.label_utils import create_label_map


def load_model(checkpoint_name):

    model = BirdCLEFModel()

    checkpoint_path = os.path.join(
        CHECKPOINT_DIR,
        checkpoint_name
    )

    model.load_state_dict(
        torch.load(
            checkpoint_path,
            map_location=DEVICE
        )
    )

    model.to(DEVICE)

    model.eval()

    return model


def main():

    # ========================================================
    # LABEL MAP
    # ========================================================

    train_csv = pd.read_csv(
        TRAIN_CSV
    )

    label_to_idx, idx_to_label = create_label_map(
        train_csv
    )

    # ========================================================
    # DATA
    # ========================================================

    df = pd.read_csv(
        "train_soundscapes_labels.csv"
    )

    dataset = SoundscapeDataset(
        df,
        label_to_idx
    )

    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0
    )

    # ========================================================
    # LOAD MODELS
    # ========================================================

    model_soundscape = load_model(
        "best_soundscape_model.pth"
    )

    model_pseudo = load_model(
        "best_pseudo_model.pth"
    )

    print(
        "\n✅ Models loaded"
    )

    # ========================================================
    # STORAGE
    # ========================================================

    all_predictions = []

    all_targets = []

    # ========================================================
    # INFERENCE
    # ========================================================

    with torch.no_grad():

        for batch in tqdm(loader):

            images = batch["image"].to(
                DEVICE
            )

            targets = batch["target"]

            # =================================================
            # MODEL 1
            # =================================================

            probs_1 = torch.sigmoid(
                model_soundscape(images)
            )

            # =================================================
            # MODEL 2
            # =================================================

            probs_2 = torch.sigmoid(
                model_pseudo(images)
            )

            # =================================================
            # ENSEMBLE
            # =================================================

            probs = (
                probs_1 + probs_2
            ) / 2

            probs = probs.cpu().numpy()

            all_predictions.append(
                probs
            )

            all_targets.append(
                targets.numpy()
            )

    # ========================================================
    # CONCAT
    # ========================================================

    all_predictions = np.concatenate(
        all_predictions
    )

    all_targets = np.concatenate(
        all_targets
    )

    print(
        "\nPrediction shape:",
        all_predictions.shape
    )

    # ========================================================
    # OPTIMIZE THRESHOLDS
    # ========================================================

    thresholds = {}

    threshold_range = np.arange(
        0.1,
        0.95,
        0.05
    )

    for class_idx in tqdm(
        range(NUM_CLASSES)
    ):

        y_true = all_targets[
            :,
            class_idx
        ]

        y_prob = all_predictions[
            :,
            class_idx
        ]

        # ====================================================
        # SKIP EMPTY CLASSES
        # ====================================================

        if y_true.sum() == 0:

            thresholds[
                idx_to_label[class_idx]
            ] = 0.5

            continue

        best_threshold = 0.5

        best_f1 = 0

        for threshold in threshold_range:

            y_pred = (
                y_prob > threshold
            ).astype(int)

            score = f1_score(
                y_true,
                y_pred,
                zero_division=0
            )

            if score > best_f1:

                best_f1 = score

                best_threshold = threshold

        thresholds[
            idx_to_label[class_idx]
        ] = float(best_threshold)

    # ========================================================
    # SAVE
    # ========================================================

    threshold_df = pd.DataFrame({

        "species":
        list(thresholds.keys()),

        "threshold":
        list(thresholds.values())
    })

    threshold_df.to_csv(
        "optimized_thresholds.csv",
        index=False
    )

    print(
        "\n✅ Threshold optimization complete"
    )

    print(
        "\nSaved:"
        " optimized_thresholds.csv"
    )


if __name__ == "__main__":

    main()