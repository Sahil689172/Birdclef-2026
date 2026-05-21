import librosa
import numpy as np

from src.configs.config import *


# ============================================================
# LOAD AUDIO
# ============================================================

def load_audio(path: str):

    y, sr = librosa.load(
        path,
        sr=SR,
        mono=True
    )

    return y


# ============================================================
# RANDOM CROP
# ============================================================

def random_crop(audio: np.ndarray):

    target_length = int(SR * DURATION)

    if len(audio) <= target_length:
        return audio

    start = np.random.randint(0, len(audio) - target_length)

    return audio[start:start + target_length]


# ============================================================
# REPEAT PADDING
# ============================================================

def repeat_pad(audio: np.ndarray):

    target_length = int(SR * DURATION)

    if len(audio) >= target_length:
        return audio

    repeat_count = int(np.ceil(target_length / len(audio)))

    audio = np.tile(audio, repeat_count)

    return audio[:target_length]


# ============================================================
# LOG MEL SPECTROGRAM
# ============================================================

def create_mel_spectrogram(audio: np.ndarray):

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX
    )

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max
    )

    return mel_db


# ============================================================
# NORMALIZATION
# ============================================================

def normalize(x: np.ndarray):

    x_min = x.min()
    x_max = x.max()

    x = 2 * (x - x_min) / (x_max - x_min + 1e-8) - 1

    return x


# ============================================================
# CREATE 3-CHANNEL TENSOR
# ============================================================

def create_3_channel_tensor(mel_db: np.ndarray):

    delta = librosa.feature.delta(mel_db)

    delta2 = librosa.feature.delta(
        mel_db,
        order=2
    )

    mel_db = normalize(mel_db)
    delta = normalize(delta)
    delta2 = normalize(delta2)

    stacked = np.stack([
        mel_db,
        delta,
        delta2
    ])

    return stacked.astype(np.float32)