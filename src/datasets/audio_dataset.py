import os

import torch
from torch.utils.data import Dataset

from src.configs.config import *
from src.utils.audio_utils import *
from src.utils.label_utils import *


class AudioDataset(Dataset):

    def __init__(
        self,
        dataframe,
        label_to_idx
    ):

        self.df = dataframe.reset_index(drop=True)

        self.label_to_idx = label_to_idx

    def __len__(self):

        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        primary_label = row["primary_label"]

        secondary_labels = row["secondary_labels"]

        filename = row["filename"]

        # ====================================================
        # AUDIO PATH
        # ====================================================

        audio_path = os.path.join(
            TRAIN_AUDIO_DIR,
            filename
        )

        # ====================================================
        # LOAD AUDIO
        # ====================================================

        audio = load_audio(audio_path)

        # ====================================================
        # RANDOM CROP
        # ====================================================

        audio = random_crop(audio)

        # ====================================================
        # REPEAT PAD
        # ====================================================

        audio = repeat_pad(audio)

        # ====================================================
        # MEL SPECTROGRAM
        # ====================================================

        mel_db = create_mel_spectrogram(audio)

        # ====================================================
        # CREATE 3-CHANNEL TENSOR
        # ====================================================

        image = create_3_channel_tensor(mel_db)

        # ====================================================
        # LABEL ENCODING
        # ====================================================

        target = create_target(
            primary_label,
            secondary_labels,
            self.label_to_idx
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