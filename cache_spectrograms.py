import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.configs.config import *
from src.utils.audio_utils import *


CACHE_DIR = "spectrogram_cache"

os.makedirs(CACHE_DIR, exist_ok=True)


def main():

    train_df = pd.read_csv(TRAIN_CSV)

    for idx, row in tqdm(
        train_df.iterrows(),
        total=len(train_df)
    ):

        filename = row["filename"]

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

        tensor = create_3_channel_tensor(mel_db)

        # ====================================================
        # SAVE AS .NPY
        # ====================================================

        save_name = filename.replace(
            "/",
            "_"
        ).replace(
            ".ogg",
            ".npy"
        )

        save_path = os.path.join(
            CACHE_DIR,
            save_name
        )

        np.save(
            save_path,
            tensor
        )

    print("\n✅ ALL SPECTROGRAMS CACHED")


if __name__ == "__main__":

    main()