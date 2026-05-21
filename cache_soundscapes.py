import os

import librosa
import numpy as np
import pandas as pd

from tqdm import tqdm

from src.configs.config import *
from src.utils.audio_utils import *


CACHE_DIR = "soundscape_cache"

os.makedirs(
    CACHE_DIR,
    exist_ok=True
)


# ============================================================
# TIME STRING → SECONDS
# ============================================================

def time_to_seconds(t):

    h, m, s = t.split(":")

    return (
        int(h) * 3600 +
        int(m) * 60 +
        int(s)
    )


def main():

    labels_df = pd.read_csv(
        "train_soundscapes_labels.csv"
    )

    for idx, row in tqdm(
        labels_df.iterrows(),
        total=len(labels_df)
    ):

        filename = row["filename"]

        start_time = time_to_seconds(
            row["start"]
        )

        end_time = time_to_seconds(
            row["end"]
        )

        # ====================================================
        # AUDIO PATH
        # ====================================================

        audio_path = os.path.join(
            "train_soundscapes",
            filename
        )

        # ====================================================
        # LOAD AUDIO
        # ====================================================

        audio, sr = librosa.load(
            audio_path,
            sr=SR,
            mono=True
        )

        # ====================================================
        # EXTRACT WINDOW
        # ====================================================

        start_sample = int(
            start_time * SR
        )

        end_sample = int(
            end_time * SR
        )

        clip = audio[
            start_sample:end_sample
        ]

        # ====================================================
        # PAD
        # ====================================================

        clip = repeat_pad(clip)

        # ====================================================
        # MEL
        # ====================================================

        mel_db = create_mel_spectrogram(
            clip
        )

        tensor = create_3_channel_tensor(
            mel_db
        )

        # ====================================================
        # SAVE
        # ====================================================

        cache_name = (
            f"{filename}_"
            f"{row['start']}_"
            f"{row['end']}.npy"
        )

        cache_name = cache_name.replace(
            ":",
            "-"
        ).replace(
            "/",
            "_"
        )

        save_path = os.path.join(
            CACHE_DIR,
            cache_name
        )

        np.save(
            save_path,
            tensor
        )

    print("\n✅ Soundscape cache complete")


if __name__ == "__main__":

    main()