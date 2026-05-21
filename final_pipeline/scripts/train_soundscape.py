import os

import pandas as pd
import torch

from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader

from src.configs.config import *
from src.datasets.soundscape_dataset import SoundscapeDataset

from src.models.efficientnet_model import BirdCLEFModel

from src.training.engine import *
from src.training.losses import get_loss
from src.training.metrics import macro_auc

from src.utils.label_utils import create_label_map


# ============================================================
# CHECKPOINTS
# ============================================================

os.makedirs(
    CHECKPOINT_DIR,
    exist_ok=True
)


def main():

    # ========================================================
    # LOAD ORIGINAL TRAIN CSV
    # ========================================================

    train_csv = pd.read_csv(
        TRAIN_CSV
    )

    # ========================================================
    # GLOBAL LABEL MAP
    # ========================================================

    label_to_idx, idx_to_label = create_label_map(
        train_csv
    )

    # ========================================================
    # LOAD SOUNDSCAPE LABELS
    # ========================================================

    df = pd.read_csv(
        "train_soundscapes_labels.csv"
    )

    print(
        f"\nTotal soundscape segments: "
        f"{len(df)}"
    )

    # ========================================================
    # SPLIT
    # ========================================================

    train_df, valid_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42
    )

    # ========================================================
    # DATASETS
    # ========================================================

    train_dataset = SoundscapeDataset(
        train_df,
        label_to_idx
    )

    valid_dataset = SoundscapeDataset(
        valid_df,
        label_to_idx
    )

    # ========================================================
    # DATALOADERS
    # ========================================================

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # ========================================================
    # MODEL
    # ========================================================

    model = BirdCLEFModel()

    # ========================================================
    # LOAD BASELINE CHECKPOINT
    # ========================================================

    checkpoint_path = os.path.join(
        CHECKPOINT_DIR,
        "best_model.pth"
    )

    model.load_state_dict(
        torch.load(
            checkpoint_path,
            map_location=DEVICE
        )
    )

    print("\n✅ Loaded baseline checkpoint")

    model.to(DEVICE)

    # ========================================================
    # LOSS
    # ========================================================

    criterion = get_loss()

    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-5,
        weight_decay=WEIGHT_DECAY
    )

    # ========================================================
    # BEST SCORE
    # ========================================================

    best_auc = 0

    # ========================================================
    # TRAIN LOOP
    # ========================================================

    for epoch in range(EPOCHS):

        print(f"\n======== EPOCH {epoch+1} ========")

        # ====================================================
        # TRAIN
        # ====================================================

        train_loss = train_fn(
            train_loader,
            model,
            optimizer,
            criterion
        )

        # ====================================================
        # VALIDATE
        # ====================================================

        valid_loss, preds, targets = valid_fn(
            valid_loader,
            model,
            criterion
        )

        # ====================================================
        # METRIC
        # ====================================================

        auc = macro_auc(
            targets,
            preds
        )

        # ====================================================
        # LOGGING
        # ====================================================

        print(f"\nTrain Loss : {train_loss:.4f}")

        print(f"Valid Loss : {valid_loss:.4f}")

        print(f"Valid AUC  : {auc:.4f}")

        # ====================================================
        # SAVE
        # ====================================================

        if auc > best_auc:

            best_auc = auc

            save_path = os.path.join(
                CHECKPOINT_DIR,
                "best_soundscape_model.pth"
            )

            torch.save(
                model.state_dict(),
                save_path
            )

            print(
                "\n✅ Best soundscape "
                "model saved"
            )

    print("\nTRAINING COMPLETED")


if __name__ == "__main__":

    main()