import os

import numpy as np
import pandas as pd
import torch

from tqdm import tqdm

from torch.utils.data import DataLoader

from src.configs.config import *

from src.datasets.soundscape_dataset import SoundscapeDataset

from src.models.efficientnet_model import BirdCLEFModel

from src.utils.label_utils import create_label_map


THRESHOLD = 0.5


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
    # ENSEMBLE STORAGE
    # ========================================================

    all_predictions = []

    # ========================================================
    # INFERENCE
    # ========================================================

    with torch.no_grad():

        for batch in tqdm(loader):

            images = batch["image"].to(
                DEVICE
            )

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
            # AVERAGE
            # =================================================

            ensemble_probs = (
                probs_1 + probs_2
            ) / 2

            ensemble_probs = ensemble_probs.cpu().numpy()

            all_predictions.append(
                ensemble_probs
            )

    # ========================================================
    # CONCAT
    # ========================================================

    all_predictions = np.concatenate(
        all_predictions
    )

    print(
        "\nEnsemble prediction shape:",
        all_predictions.shape
    )

    # ========================================================
    # SIMPLE INSPECTION
    # ========================================================

    print(
        "\nMax probability:",
        all_predictions.max()
    )

    print(
        "Mean probability:",
        all_predictions.mean()
    )

    print(
        "\n✅ Ensemble inference completed"
    )


if __name__ == "__main__":

    main()