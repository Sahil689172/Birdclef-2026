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


THRESHOLD = 0.70


def main():

    # ========================================================
    # ORIGINAL LABEL MAP
    # ========================================================

    train_csv = pd.read_csv(
        TRAIN_CSV
    )

    label_to_idx, idx_to_label = create_label_map(
        train_csv
    )

    # ========================================================
    # SOUNDSCAPE LABELS
    # ========================================================

    df = pd.read_csv(
        "train_soundscapes_labels.csv"
    )

    # ========================================================
    # DATASET
    # ========================================================

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
    # MODEL
    # ========================================================

    model = BirdCLEFModel()

    checkpoint_path = os.path.join(
        CHECKPOINT_DIR,
        "best_soundscape_model.pth"
    )

    model.load_state_dict(
        torch.load(
            checkpoint_path,
            map_location=DEVICE
        )
    )

    model.to(DEVICE)

    model.eval()

    # ========================================================
    # PSEUDO LABEL STORAGE
    # ========================================================

    pseudo_rows = []

    # ========================================================
    # INFERENCE
    # ========================================================

    with torch.no_grad():

        sample_idx = 0

        for batch in tqdm(loader):

            images = batch["image"].to(
                DEVICE
            )

            outputs = model(images)

            probs = torch.sigmoid(
                outputs
            )

            probs = probs.cpu().numpy()
            print(
    "Max prob in batch:",
    probs.max()
)
            for pred in probs:

                predicted_species = []

                for i, score in enumerate(pred):

                    if score > THRESHOLD:

                        predicted_species.append(
                            idx_to_label[i]
                        )

                # ============================================
                # ONLY KEEP CONFIDENT PREDICTIONS
                # ============================================

                if len(predicted_species) > 0:

                    row = df.iloc[sample_idx]

                    pseudo_rows.append({

                        "filename":
                        row["filename"],

                        "start":
                        row["start"],

                        "end":
                        row["end"],

                        "pseudo_labels":
                        ";".join(
                            predicted_species
                        )
                    })

                sample_idx += 1

    # ========================================================
    # SAVE
    # ========================================================

    pseudo_df = pd.DataFrame(
        pseudo_rows
    )

    pseudo_df.to_csv(
        "pseudo_labels.csv",
        index=False
    )

    print(
        f"\n✅ Generated "
        f"{len(pseudo_df)} "
        f"pseudo labels"
    )


if __name__ == "__main__":

    main()