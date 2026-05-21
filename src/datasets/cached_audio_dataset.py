import os
import random

import numpy as np
import torch

from torch.utils.data import Dataset

from src.configs.config import *
from src.utils.label_utils import *
from src.utils.augmentations import *


class CachedAudioDataset(Dataset):

    def __init__(
        self,
        dataframe,
        label_to_idx,
        train=True
    ):

        self.df = dataframe.reset_index(drop=True)

        self.label_to_idx = label_to_idx

        self.train = train

    def __len__(self):

        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        primary_label = row["primary_label"]

        secondary_labels = row["secondary_labels"]

        filename = row["filename"]

        # ====================================================
        # CACHE PATH
        # ====================================================

        cache_name = filename.replace(
            "/",
            "_"
        ).replace(
            ".ogg",
            ".npy"
        )

        cache_path = os.path.join(
            "spectrogram_cache",
            cache_name
        )

        # ====================================================
        # LOAD CACHED TENSOR
        # ====================================================

        image = np.load(cache_path)

        # ====================================================
        # LABEL ENCODING
        # ====================================================

        target = create_target(
            primary_label,
            secondary_labels,
            self.label_to_idx
        )

        # ====================================================
        # TRAIN AUGMENTATIONS
        # ====================================================

        if self.train:

            # ================================================
            # RANDOM TIME SHIFT
            # Mild temporal robustness
            # ================================================

            if random.random() < 0.25:

                image = random_time_shift(
                    image,
                    max_shift=10
                )

            # ================================================
            # RANDOM GAIN
            # Simulate volume variation
            # ================================================

            if random.random() < 0.4:

                image = random_gain(
                    image,
                    min_gain=0.9,
                    max_gain=1.1
                )

            # ================================================
            # SPEC AUGMENT
            # Reduced masking strength
            # ================================================

            if random.random() < 0.25:

                image = spec_augment(
                    image,
                    time_mask=15,
                    freq_mask=8
                )

            # ================================================
            # MIXUP
            # Light Mixup
            # ================================================

            if random.random() < 0.1:

                mix_idx = random.randint(
                    0,
                    len(self.df) - 1
                )

                mix_row = self.df.iloc[mix_idx]

                mix_filename = mix_row["filename"]

                mix_cache_name = mix_filename.replace(
                    "/",
                    "_"
                ).replace(
                    ".ogg",
                    ".npy"
                )

                mix_cache_path = os.path.join(
                    "spectrogram_cache",
                    mix_cache_name
                )

                mix_image = np.load(
                    mix_cache_path
                )

                mix_target = create_target(
                    mix_row["primary_label"],
                    mix_row["secondary_labels"],
                    self.label_to_idx
                )

                image, target = mixup(
                    image,
                    target,
                    mix_image,
                    mix_target,
                    alpha=0.2
                )

        # ====================================================
        # TORCH TENSORS
        # ====================================================

        image = torch.tensor(
            image,
            dtype=torch.float32
        )

        target = torch.tensor(
            target,
            dtype=torch.float32
        )

        return {
            "image": image,
            "target": target
        }