import os

import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset

from src.configs.config import *


class PseudoDataset(Dataset):

    def __init__(
        self,
        dataframe,
        label_to_idx
    ):

        self.df = dataframe.reset_index(
            drop=True
        )

        self.label_to_idx = label_to_idx

    def __len__(self):

        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        filename = row["filename"]

        start = row["start"]

        end = row["end"]

        labels = row["labels"]

        # ====================================================
        # CACHE NAME
        # ====================================================

        cache_name = (
            f"{filename}_{start}_{end}.npy"
        )

        cache_name = cache_name.replace(
            ":",
            "-"
        ).replace(
            "/",
            "_"
        )

        cache_path = os.path.join(
            "soundscape_cache",
            cache_name
        )

        # ====================================================
        # LOAD IMAGE
        # ====================================================

        image = np.load(cache_path)

        # ====================================================
        # TARGET VECTOR
        # ====================================================

        target = np.zeros(
            NUM_CLASSES,
            dtype=np.float32
        )

        label_list = str(labels).split(";")

        for species in label_list:

            if species in self.label_to_idx:

                target[
                    self.label_to_idx[species]
                ] = 1.0

        # ====================================================
        # TORCH
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