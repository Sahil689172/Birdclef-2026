import os

import pandas as pd
import torch

from sklearn.model_selection import GroupShuffleSplit

from torch.utils.data import DataLoader

from src.configs.config import *

from src.datasets.pseudo_dataset import PseudoDataset

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
    # LABEL MAP
    # ========================================================

    train_csv = pd.read_csv(
        TRAIN_CSV
    )

    label_to_idx, idx_to_label = create_label_map(
        train_csv
    )

    # ========================================================
    # ORIGINAL SOUNDSCAPE LABELS
    # ========================================================

    original_df = pd.read_csv(
        "train_soundscapes_labels.csv"
    )

    original_df = original_df.rename(
        columns={
            "primary_label": "labels"
        }
    )

    # ========================================================
    # PSEUDO LABELS
    # ========================================================

    pseudo_df = pd.read_csv(
        "pseudo_labels.csv"
    )

    pseudo_df = pseudo_df.rename(
        columns={
            "pseudo_labels": "labels"
        }
    )

    # ========================================================
    # MERGE DATASETS
    # ========================================================

    full_df = pd.concat(
        [
            original_df[
                [
                    "filename",
                    "start",
                    "end",
                    "labels"
                ]
            ],

            pseudo_df[
                [
                    "filename",
                    "start",
                    "end",
                    "labels"
                ]
            ]
        ],
        ignore_index=True
    )

    print(
        f"\nOriginal labels: "
        f"{len(original_df)}"
    )

    print(
        f"Pseudo labels: "
        f"{len(pseudo_df)}"
    )

    print(
        f"Combined dataset: "
        f"{len(full_df)}"
    )

    # ========================================================
    # GROUP SPLIT
    # ========================================================

    gss = GroupShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=42
    )

    groups = full_df["filename"]

    train_idx, valid_idx = next(
        gss.split(
            full_df,
            groups=groups
        )
    )

    train_df = full_df.iloc[
        train_idx
    ]

    valid_df = full_df.iloc[
        valid_idx
    ]

    print(
        f"\nTrain size: "
        f"{len(train_df)}"
    )

    print(
        f"Valid size: "
        f"{len(valid_df)}"
    )

    # ========================================================
    # DATASETS
    # ========================================================

    train_dataset = PseudoDataset(
        train_df,
        label_to_idx
    )

    valid_dataset = PseudoDataset(
        valid_df,
        label_to_idx
    )

    # ========================================================
    # LOADERS
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

    print(
        "\n✅ Loaded soundscape model"
    )

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

        print(
            f"\n======== EPOCH "
            f"{epoch+1} ========"
        )

        train_loss = train_fn(
            train_loader,
            model,
            optimizer,
            criterion
        )

        valid_loss, preds, targets = valid_fn(
            valid_loader,
            model,
            criterion
        )

        auc = macro_auc(
            targets,
            preds
        )

        print(
            f"\nTrain Loss : "
            f"{train_loss:.4f}"
        )

        print(
            f"Valid Loss : "
            f"{valid_loss:.4f}"
        )

        print(
            f"Valid AUC  : "
            f"{auc:.4f}"
        )

        # ====================================================
        # SAVE
        # ====================================================

        if auc > best_auc:

            best_auc = auc

            save_path = os.path.join(
                CHECKPOINT_DIR,
                "best_pseudo_model.pth"
            )

            torch.save(
                model.state_dict(),
                save_path
            )

            print(
                "\n✅ Best pseudo "
                "model saved"
            )

    print("\nTRAINING COMPLETED")


if __name__ == "__main__":

    main()