import random
import numpy as np


# ============================================================
# SPEC AUGMENT
# ============================================================

def spec_augment(
    image,
    time_mask=15,
    freq_mask=8
):

    augmented = image.copy()

    _, freq_bins, time_bins = augmented.shape

    # ========================================================
    # TIME MASK
    # ========================================================

    t = random.randint(0, time_mask)

    t0 = random.randint(0, max(0, time_bins - t))

    augmented[:, :, t0:t0+t] = 0

    # ========================================================
    # FREQUENCY MASK
    # ========================================================

    f = random.randint(0, freq_mask)

    f0 = random.randint(0, max(0, freq_bins - f))

    augmented[:, f0:f0+f, :] = 0

    return augmented


# ============================================================
# RANDOM TIME SHIFT
# ============================================================

def random_time_shift(
    image,
    max_shift=20
):

    shift = random.randint(
        -max_shift,
        max_shift
    )

    return np.roll(
        image,
        shift,
        axis=2
    )


# ============================================================
# RANDOM GAIN
# ============================================================

def random_gain(
    image,
    min_gain=0.8,
    max_gain=1.2
):

    gain = random.uniform(
        min_gain,
        max_gain
    )

    image = image * gain

    image = np.clip(
        image,
        -1,
        1
    )

    return image


# ============================================================
# MIXUP
# ============================================================

def mixup(
    image1,
    target1,
    image2,
    target2,
    alpha=0.2
):

    lam = np.random.beta(alpha, alpha)

    mixed_image = (
        lam * image1 +
        (1 - lam) * image2
    )

    mixed_target = (
        lam * target1 +
        (1 - lam) * target2
    )

    return mixed_image, mixed_target
    