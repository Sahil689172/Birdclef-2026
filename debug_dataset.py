import pandas as pd
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from src.datasets.audio_dataset import AudioDataset
from src.configs.config import *


def main():

    # ========================================================
    # LOAD CSV
    # ========================================================

    train_df = pd.read_csv(TRAIN_CSV)

    # ========================================================
    # DATASET
    # ========================================================

    dataset = AudioDataset(train_df)

    print(f"\nDataset size: {len(dataset)}")

    # ========================================================
    # SINGLE SAMPLE
    # ========================================================

    sample = dataset[0]

    image = sample["image"]
    target = sample["target"]

    print("\nImage shape:")
    print(image.shape)

    print("\nTarget shape:")
    print(target.shape)

    print("\nImage range:")
    print(image.min(), image.max())

    print("\nPositive labels:")
    print((target > 0).sum())

    # ========================================================
    # VISUALIZE MEL CHANNEL
    # ========================================================

    plt.figure(figsize=(12, 4))

    plt.imshow(
        image[0],
        aspect="auto",
        origin="lower"
    )

    plt.title("Mel Spectrogram")

    plt.colorbar()

    plt.tight_layout()

    plt.show()

    # ========================================================
    # DATALOADER
    # ========================================================

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,   # IMPORTANT FOR WINDOWS
        pin_memory=False
    )

    batch = next(iter(loader))

    print("\nBatch image shape:")
    print(batch["image"].shape)

    print("\nBatch target shape:")
    print(batch["target"].shape)

    print("\nPHASE 2 PREPROCESSING PIPELINE WORKING SUCCESSFULLY")


# ============================================================
# WINDOWS SAFE ENTRY
# ============================================================

if __name__ == "__main__":

    main()