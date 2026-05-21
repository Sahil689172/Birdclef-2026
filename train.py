import os

import pandas as pd
import torch

from sklearn.model_selection import GroupKFold

from torch.utils.data import DataLoader

from src.configs.config import *
from src.datasets.cached_audio_dataset import CachedAudioDataset
from src.models.efficientnet_model import BirdCLEFModel

from src.training.engine import *
from src.training.losses import get_loss
from src.training.metrics import macro_auc

from src.utils.label_utils import create_label_map


# ============================================================
# CHECKPOINT DIRECTORY
# ============================================================

os.makedirs(
    CHECKPOINT_DIR,
    exist_ok=True
)


def main():

    # ========================================================
    # LOAD DATA
    # ========================================================

    train_df = pd.read_csv(TRAIN_CSV)

    # ========================================================
    # GLOBAL LABEL MAP
    # ========================================================

    label_to_idx, idx_to_label = create_label_map(
        train_df
    )

    # ========================================================
    # GROUP KFOLD
    # ========================================================

    gkf = GroupKFold(n_splits=5)

    groups = train_df["primary_label"]

    train_idx, valid_idx = next(
        gkf.split(
            train_df,
            groups=groups
        )
    )

    train_fold = train_df.iloc[train_idx]

    valid_fold = train_df.iloc[valid_idx]

    print(f"\nTrain size: {len(train_fold)}")

    print(f"Valid size: {len(valid_fold)}")

    # ========================================================
    # DATASETS
    # ========================================================

    train_dataset = CachedAudioDataset(
        train_fold,
        label_to_idx,
        train=True
    )

    valid_dataset = CachedAudioDataset(
        valid_fold,
        label_to_idx,
        train=False
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
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    # ========================================================
    # BEST SCORE
    # ========================================================

    best_auc = 0

    # ========================================================
    # TRAINING LOOP
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
        # SAVE BEST MODEL
        # ====================================================

        if auc > best_auc:

            best_auc = auc

            save_path = os.path.join(
                CHECKPOINT_DIR,
                "best_model.pth"
            )

            torch.save(
                model.state_dict(),
                save_path
            )

            print("\n✅ Best model saved")

    print("\nTRAINING COMPLETED")


if __name__ == "__main__":

    main()