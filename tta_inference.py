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


# ============================================================
# TTA FUNCTIONS
# ============================================================

def time_shift(image, shift=10):

    return np.roll(
        image,
        shift,
        axis=2
    )


def gain_adjust(image, gain=1.05):

    image = image * gain

    image = np.clip(
        image,
        -1,
        1
    )

    return image


# ============================================================
# LOAD MODEL
# ============================================================

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


def predict_with_tta(
    model,
    images
):

    predictions = []

    # ========================================================
    # ORIGINAL
    # ========================================================

    preds = torch.sigmoid(
        model(images)
    )

    predictions.append(preds)

    # ========================================================
    # TIME SHIFT
    # ========================================================

    shifted = []

    for img in images.cpu().numpy():

        shifted_img = time_shift(
            img,
            shift=10
        )

        shifted.append(
            shifted_img
        )

    shifted = torch.tensor(
        np.array(shifted),
        dtype=torch.float32
    ).to(DEVICE)

    preds_shift = torch.sigmoid(
        model(shifted)
    )

    predictions.append(
        preds_shift
    )

    # ========================================================
    # GAIN ADJUST
    # ========================================================

    gained = []

    for img in images.cpu().numpy():

        gained_img = gain_adjust(
            img,
            gain=1.05
        )

        gained.append(
            gained_img
        )

    gained = torch.tensor(
        np.array(gained),
        dtype=torch.float32
    ).to(DEVICE)

    preds_gain = torch.sigmoid(
        model(gained)
    )

    predictions.append(
        preds_gain
    )

    # ========================================================
    # AVERAGE
    # ========================================================

    final_preds = torch.stack(
        predictions
    ).mean(dim=0)

    return final_preds


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
    # LOAD MODEL
    # ========================================================

    model = load_model(
        "best_pseudo_model.pth"
    )

    print(
        "\n✅ Model loaded"
    )

    # ========================================================
    # STORAGE
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

            preds = predict_with_tta(
                model,
                images
            )

            preds = preds.cpu().numpy()

            all_predictions.append(
                preds
            )

    # ========================================================
    # CONCAT
    # ========================================================

    all_predictions = np.concatenate(
        all_predictions
    )

    print(
        "\nPrediction shape:",
        all_predictions.shape
    )

    print(
        "\nMax probability:",
        all_predictions.max()
    )

    print(
        "Mean probability:",
        all_predictions.mean()
    )

    print(
        "\n✅ TTA inference complete"
    )


if __name__ == "__main__":

    main()