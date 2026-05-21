import os

import librosa
import numpy as np
import pandas as pd
import torch

from tqdm import tqdm

from src.configs.config import *

from src.models.efficientnet_model import BirdCLEFModel

from src.utils.audio_utils import (
    create_mel_spectrogram,
    create_3_channel_tensor
)

from src.utils.label_utils import create_label_map


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


# ============================================================
# TTA
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
# PREDICT WITH TTA
# ============================================================

def predict_tta(model, image):

    predictions = []

    # ========================================================
    # ORIGINAL
    # ========================================================

    x = torch.tensor(
        image,
        dtype=torch.float32
    ).unsqueeze(0).to(DEVICE)

    preds = torch.sigmoid(
        model(x)
    )

    predictions.append(preds)

    # ========================================================
    # SHIFT
    # ========================================================

    shifted = time_shift(
        image,
        shift=10
    )

    x_shift = torch.tensor(
        shifted,
        dtype=torch.float32
    ).unsqueeze(0).to(DEVICE)

    preds_shift = torch.sigmoid(
        model(x_shift)
    )

    predictions.append(
        preds_shift
    )

    # ========================================================
    # GAIN
    # ========================================================

    gained = gain_adjust(
        image,
        gain=1.05
    )

    x_gain = torch.tensor(
        gained,
        dtype=torch.float32
    ).unsqueeze(0).to(DEVICE)

    preds_gain = torch.sigmoid(
        model(x_gain)
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

    return final_preds.squeeze(0)


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
    # LOAD THRESHOLDS
    # ========================================================

    threshold_df = pd.read_csv(
        "optimized_thresholds.csv"
    )

    threshold_map = dict(

        zip(
            threshold_df["species"],
            threshold_df["threshold"]
        )
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
    # TEST FILES
    # ========================================================

    test_dir = "test_soundscapes"

    test_files = [

        f for f in os.listdir(test_dir)

        if f.endswith(".ogg")
    ]

    print(
        f"\nFound "
        f"{len(test_files)} "
        f"test files"
    )

    # ========================================================
    # SUBMISSION STORAGE
    # ========================================================

    submission_rows = []

    # ========================================================
    # PROCESS FILES
    # ========================================================

    for filename in tqdm(test_files):

        filepath = os.path.join(
            test_dir,
            filename
        )

        # ====================================================
        # LOAD AUDIO
        # ====================================================

        audio, sr = librosa.load(
            filepath,
            sr=SR,
            mono=True
        )

        # ====================================================
        # WINDOWING
        # ====================================================

        total_samples = len(audio)

        window_size = int(
            SR * DURATION
        )

        num_windows = total_samples // window_size

        # ====================================================
        # WINDOW LOOP
        # ====================================================

        for i in range(num_windows):

            start = i * window_size

            end = start + window_size

            clip = audio[
                start:end
            ]

            # ================================================
            # MEL
            # ================================================

            mel = create_mel_spectrogram(
                clip
            )

            image = create_3_channel_tensor(
                mel
            )

            # ================================================
            # MODEL 1
            # ================================================

            probs_1 = predict_tta(
                model_soundscape,
                image
            )

            # ================================================
            # MODEL 2
            # ================================================

            probs_2 = predict_tta(
                model_pseudo,
                image
            )

            # ================================================
            # ENSEMBLE
            # ================================================

            probs = (
                probs_1 + probs_2
            ) / 2

            probs = probs.detach().cpu().numpy()

            # ================================================
            # ROW ID
            # ================================================

            row_id = (
                f"{filename}_"
                f"{i*5}"
            )

            row = {

                "row_id":
                row_id
            }

            # ================================================
            # THRESHOLDS
            # ================================================

            for species in idx_to_label.values():

                class_idx = label_to_idx[
                    species
                ]

                threshold = threshold_map.get(
                    species,
                    0.5
                )

                score = probs[
                    class_idx
                ]

                # ============================================
                # STORE PROBABILITY
                # ============================================

                row[species] = float(score)

            submission_rows.append(
                row
            )

    # ========================================================
    # CREATE SUBMISSION
    # ========================================================

    submission_df = pd.DataFrame(
        submission_rows
    )

    submission_df.to_csv(
        "submission.csv",
        index=False
    )

    print(
        "\n✅ submission.csv generated"
    )


if __name__ == "__main__":

    main()
