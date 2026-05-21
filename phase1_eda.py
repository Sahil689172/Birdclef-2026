import os
import random
import warnings
from collections import Counter
from itertools import combinations

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import soundfile as sf
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================

TRAIN_CSV             = "train.csv"
TAXONOMY_CSV          = "taxonomy.csv"
SOUNDSCAPE_LABELS_CSV = "train_soundscapes_labels.csv"
TRAIN_AUDIO_DIR       = "train_audio"

OUTPUT_DIR  = "outputs"
PLOTS_DIR   = os.path.join(OUTPUT_DIR, "plots")
REPORTS_DIR = os.path.join(OUTPUT_DIR, "reports")

os.makedirs(PLOTS_DIR,   exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

SR            = 32000
DURATION      = 5.0     # fixed 5-sec window — matches submission format
N_MELS        = 128
N_FFT         = 1024
HOP_LENGTH    = 512
EXPECTED_COLS = 313     # time frames for exactly 5 sec at SR=32000, hop=512

# ============================================================
# STEP 1 — LOAD DATASETS
# ============================================================

print("\n" + "=" * 60)
print("STEP 1 — LOADING CSV FILES")
print("=" * 60)

train_df    = pd.read_csv(TRAIN_CSV)
taxonomy_df = pd.read_csv(TAXONOMY_CSV)
sc_df       = pd.read_csv(SOUNDSCAPE_LABELS_CSV)

print(f"train.csv shape        : {train_df.shape}")
print(f"taxonomy.csv shape     : {taxonomy_df.shape}")
print(f"soundscape_labels shape: {sc_df.shape}")

# ============================================================
# STEP 2 — BASIC DATASET ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("STEP 2 — BASIC DATASET ANALYSIS")
print("=" * 60)

print("\nMissing values:")
print(train_df.isnull().sum())

print(f"\nUnique species (train.csv)  : {train_df['primary_label'].nunique()}")
print(f"Unique species (taxonomy)   : {taxonomy_df['primary_label'].nunique()}")
print(f"Duplicate rows              : {train_df.duplicated().sum()}")

# ============================================================
# STEP 3 — CLASS DISTRIBUTION ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("STEP 3 — CLASS DISTRIBUTION ANALYSIS")
print("=" * 60)

label_counts = train_df["primary_label"].value_counts()

rare_10 = (label_counts < 10).sum()
rare_20 = (label_counts < 20).sum()
rare_50 = (label_counts < 50).sum()

print(f"Species with <10 samples : {rare_10}")
print(f"Species with <20 samples : {rare_20}")
print(f"Species with <50 samples : {rare_50}")
print(f"Most common species      : {label_counts.index[0]} ({label_counts.iloc[0]} samples)")
print(f"Least common species     : {label_counts.index[-1]} ({label_counts.iloc[-1]} samples)")

# Plot — colour-code rare species in red
colors = ["#e74c3c" if c < 20 else "#3498db" for c in label_counts.sort_values(ascending=False)]

fig, ax = plt.subplots(figsize=(20, 6))
ax.bar(range(len(label_counts)), label_counts.sort_values(ascending=False), color=colors, width=1.0)
ax.set_yscale("log")
ax.set_title("Species Frequency Distribution (red = <20 samples)", fontsize=14)
ax.set_xlabel("Species (sorted by frequency)")
ax.set_ylabel("Sample count (log scale)")
ax.axhline(y=20, color="red",    linestyle="--", linewidth=1, label="20-sample threshold")
ax.axhline(y=10, color="orange", linestyle="--", linewidth=1, label="10-sample threshold")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "class_distribution.png"), dpi=150)
plt.close()
print("Saved class_distribution.png")

# ============================================================
# STEP 4 — ZERO-SHOT SPECIES ANALYSIS  ← NEW
# ============================================================

print("\n" + "=" * 60)
print("STEP 4 — ZERO-SHOT SPECIES ANALYSIS")
print("=" * 60)

taxonomy_labels = set(taxonomy_df["primary_label"].str.strip())
train_labels    = set(train_df["primary_label"].str.strip())

zero_shot   = sorted(taxonomy_labels - train_labels)
train_only  = sorted(train_labels - taxonomy_labels)

print(f"Species in taxonomy but NOT in train.csv : {len(zero_shot)}")
print(f"Species in train.csv  but NOT in taxonomy : {len(train_only)}")

if zero_shot:
    print("\nZero-shot species (need special handling):")
    for s in zero_shot:
        row = taxonomy_df[taxonomy_df["primary_label"] == s]
        if not row.empty:
            name = row["common_name"].values[0] if "common_name" in row.columns else "unknown"
            print(f"  {s:15s}  →  {name}")
        else:
            print(f"  {s}")

# Save zero-shot list
with open(os.path.join(REPORTS_DIR, "zero_shot_species.txt"), "w") as f:
    f.write("Species in taxonomy but with zero training samples:\n\n")
    for s in zero_shot:
        f.write(s + "\n")
print(f"\nSaved zero_shot_species.txt ({len(zero_shot)} species)")

# ============================================================
# STEP 5 — SECONDARY LABEL ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("STEP 5 — SECONDARY LABEL ANALYSIS")
print("=" * 60)

secondary_counter = Counter()
pair_counter      = Counter()
secondary_exists  = 0

for raw in train_df["secondary_labels"].fillna("[]"):
    try:
        labels = eval(raw) if isinstance(raw, str) else []
        if len(labels) > 0:
            secondary_exists += 1
        for lbl in labels:
            secondary_counter[lbl] += 1
        for pair in combinations(sorted(labels), 2):
            pair_counter[pair] += 1
    except Exception:
        continue

print(f"Recordings with secondary labels : {secondary_exists}")
print(f"Unique secondary species         : {len(secondary_counter)}")

print("\nTop 10 secondary labels:")
for lbl, cnt in secondary_counter.most_common(10):
    is_zs = "⚠ zero-shot" if lbl in zero_shot else ""
    print(f"  {lbl:15s}  {cnt:4d}  {is_zs}")

print("\nTop 10 co-occurring pairs:")
for pair, cnt in pair_counter.most_common(10):
    print(f"  {pair[0]:15s} + {pair[1]:15s}  {cnt:3d}")

# ============================================================
# STEP 6 — AUDIO FILE VALIDATION
# ============================================================

print("\n" + "=" * 60)
print("STEP 6 — AUDIO VALIDATION")
print("=" * 60)

audio_paths = []
for species_folder in os.listdir(TRAIN_AUDIO_DIR):
    species_path = os.path.join(TRAIN_AUDIO_DIR, species_folder)
    if not os.path.isdir(species_path):
        continue
    for fname in os.listdir(species_path):
        if fname.endswith(".ogg"):
            audio_paths.append(os.path.join(species_path, fname))

print(f"Total audio files found : {len(audio_paths)}")

durations        = []
corrupted_files  = []
very_short_files = []   # < 1 sec  → discard candidates
short_files      = []   # 1–2 sec  → repeat-pad to 5 sec
long_files       = []   # > 600 sec → flag as suspicious

for path in tqdm(audio_paths, desc="Scanning files"):
    try:
        info     = sf.info(path)
        duration = info.duration
        durations.append((path, duration))

        if duration < 1.0:
            very_short_files.append((path, duration))
        elif duration < 2.0:
            short_files.append((path, duration))

        if duration > 600:
            long_files.append((path, duration))

    except Exception:
        corrupted_files.append(path)

dur_values = np.array([d for _, d in durations])

print(f"\nCorrupted files         : {len(corrupted_files)}")
print(f"Files < 1 sec           : {len(very_short_files)}  → discard candidates")
print(f"Files 1–2 sec           : {len(short_files)}  → repeat-pad to 5 sec")
print(f"Files > 600 sec         : {len(long_files)}  → suspicious long files")

# Save reports
def save_list(filepath, items, header=""):
    with open(filepath, "w") as f:
        if header:
            f.write(header + "\n\n")
        for item in items:
            f.write(str(item) + "\n")

save_list(os.path.join(REPORTS_DIR, "corrupted_files.txt"),
          corrupted_files, "Corrupted audio files:")

save_list(os.path.join(REPORTS_DIR, "very_short_files.txt"),
          very_short_files, "Files shorter than 1 second (discard candidates):")

save_list(os.path.join(REPORTS_DIR, "short_files.txt"),
          short_files, "Files 1–2 seconds (repeat-pad in training):")

save_list(os.path.join(REPORTS_DIR, "long_files.txt"),
          long_files, "Files longer than 600 seconds (suspicious):")

if long_files:
    print("\nSuspiciously long files:")
    for path, dur in sorted(long_files, key=lambda x: -x[1]):
        print(f"  {dur:8.1f} sec  →  {path}")

# ============================================================
# STEP 7 — DURATION ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("STEP 7 — AUDIO DURATION ANALYSIS")
print("=" * 60)

print(f"Average duration  : {dur_values.mean():.2f} sec")
print(f"Median duration   : {np.median(dur_values):.2f} sec")
print(f"Shortest clip     : {dur_values.min():.4f} sec")
print(f"Longest clip      : {dur_values.max():.2f} sec")
print(f"Std deviation     : {dur_values.std():.2f} sec")

# Clip to 120s for readable histogram (outliers distort it)
clipped = dur_values[dur_values <= 120]

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

axes[0].hist(clipped, bins=100, color="#3498db", edgecolor="none")
axes[0].axvline(x=5,  color="red",    linestyle="--", label="5 sec (target window)")
axes[0].axvline(x=2,  color="orange", linestyle="--", label="2 sec (short threshold)")
axes[0].set_title("Duration Distribution (≤120 sec clips)")
axes[0].set_xlabel("Duration (seconds)")
axes[0].set_ylabel("Count")
axes[0].legend()

axes[1].hist(dur_values, bins=100, color="#2ecc71", edgecolor="none")
axes[1].set_title("Duration Distribution (all clips, including outliers)")
axes[1].set_xlabel("Duration (seconds)")
axes[1].set_ylabel("Count")

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "duration_distribution.png"), dpi=150)
plt.close()
print("Saved duration_distribution.png")

# ============================================================
# STEP 8 — RANDOM AUDIO INSPECTION
# ============================================================

print("\n" + "=" * 60)
print("STEP 8 — RANDOM AUDIO INSPECTION")
print("=" * 60)

# Only pick files with enough audio for a clean 5-sec crop
valid_paths = [p for p, d in durations if d >= DURATION]
sample_files = random.sample(valid_paths, min(5, len(valid_paths)))

fig, axes = plt.subplots(5, 1, figsize=(14, 20))

for idx, path in enumerate(sample_files):
    print(f"  Processing sample {idx + 1}: {os.path.basename(path)}")

    # Load exactly 5 seconds
    y, sr = librosa.load(path, sr=SR, duration=DURATION)

    mel    = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT,
                                             hop_length=HOP_LENGTH, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    img = librosa.display.specshow(mel_db, sr=sr, hop_length=HOP_LENGTH,
                                    x_axis="time", y_axis="mel", ax=axes[idx])
    fig.colorbar(img, ax=axes[idx], format="%+2.0f dB")

    species = path.split(os.sep)[-2]
    axes[idx].set_title(f"Sample {idx + 1} — species: {species}", fontsize=11)

plt.suptitle("Random Mel Spectrograms (5-second crops)", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "sample_spectrograms.png"), dpi=150,
            bbox_inches="tight")
plt.close()
print("Saved sample_spectrograms.png")

# ============================================================
# STEP 9 — SPECTROGRAM SHAPE VERIFICATION  ← FIXED
# ============================================================

print("\n" + "=" * 60)
print("STEP 9 — SPECTROGRAM SHAPE VERIFICATION")
print("=" * 60)

test_path = sample_files[0]

# FIX: load exactly DURATION seconds so shape is deterministic
y, sr = librosa.load(test_path, sr=SR, duration=DURATION)

# Pad if the file is shorter than DURATION (shouldn't happen here
# because we filtered valid_paths, but defensive coding)
target_samples = int(SR * DURATION)
if len(y) < target_samples:
    repeat_times = int(np.ceil(target_samples / len(y)))
    y = np.tile(y, repeat_times)[:target_samples]

mel    = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT,
                                         hop_length=HOP_LENGTH, n_mels=N_MELS)
mel_db = librosa.power_to_db(mel, ref=np.max)
delta  = librosa.feature.delta(mel_db)
delta2 = librosa.feature.delta(mel_db, order=2)

# Normalize each channel to [-1, 1] independently
def normalize(x):
    mn, mx = x.min(), x.max()
    return 2 * (x - mn) / (mx - mn + 1e-8) - 1

mel_norm    = normalize(mel_db)
delta_norm  = normalize(delta)
delta2_norm = normalize(delta2)

stacked = np.stack([mel_norm, delta_norm, delta2_norm])

print(f"Mel shape          : {mel_db.shape}")
print(f"Delta shape        : {delta.shape}")
print(f"Delta² shape       : {delta2.shape}")
print(f"Final tensor shape : {stacked.shape}")
print(f"Expected shape     : (3, {N_MELS}, {EXPECTED_COLS})")

shape_ok = stacked.shape == (3, N_MELS, EXPECTED_COLS)
print(f"Shape correct      : {'✅ YES' if shape_ok else '❌ NO — check HOP_LENGTH or DURATION'}")
print(f"Value range        : [{stacked.min():.4f}, {stacked.max():.4f}]  (should be [-1, 1])")

# ============================================================
# STEP 10 — GEOGRAPHIC ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("STEP 10 — GEOGRAPHIC ANALYSIS")
print("=" * 60)

geo_df = train_df.dropna(subset=["latitude", "longitude"])

fig, ax = plt.subplots(figsize=(10, 8))

scatter = ax.scatter(
    geo_df["longitude"],
    geo_df["latitude"],
    c=geo_df["primary_label"].astype("category").cat.codes,
    alpha=0.4,
    s=8,
    cmap="tab20"
)

# Draw Pantanal bounding box from competition metadata
pantanal_lon = [-57.6, -55.9, -55.9, -57.6, -57.6]
pantanal_lat = [-21.6, -21.6, -16.5, -16.5, -21.6]
ax.plot(pantanal_lon, pantanal_lat, "r--", linewidth=1.5, label="Pantanal region")

ax.set_title("Recording Locations by Species")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "geographic_distribution.png"), dpi=150)
plt.close()
print("Saved geographic_distribution.png")

# ============================================================
# STEP 11 — SOUNDSCAPE LABEL DENSITY
# ============================================================

print("\n" + "=" * 60)
print("STEP 11 — SOUNDSCAPE LABEL DENSITY")
print("=" * 60)

# Inspect actual column names first (defence against schema variation)
print(f"Soundscape CSV columns: {list(sc_df.columns)}")

# Try both common column-name patterns
label_col = None
for candidate in ["primary_label", "birds", "labels", "species"]:
    if candidate in sc_df.columns:
        label_col = candidate
        break

if label_col is None:
    print("Could not find label column — printing first few rows for inspection:")
    print(sc_df.head())
else:
    species_per_segment = []

    for val in sc_df[label_col].dropna():
        # Handle both semicolon-separated and list-style values
        val = str(val).strip()
        if val in ("", "[]", "nocall"):
            species_per_segment.append(0)
            continue
        parts = [p.strip() for p in val.replace("[", "").replace("]", "")
                                        .replace("'", "").split(";") if p.strip()]
        species_per_segment.append(len(parts))

    species_per_segment = np.array(species_per_segment)

    print(f"Average species per segment : {species_per_segment.mean():.2f}")
    print(f"Maximum species in segment  : {species_per_segment.max()}")
    print(f"Segments with 0 species     : {(species_per_segment == 0).sum()}")
    print(f"Segments with 1 species     : {(species_per_segment == 1).sum()}")
    print(f"Segments with 2+ species    : {(species_per_segment >= 2).sum()}")

    fig, ax = plt.subplots(figsize=(10, 5))
    counts = Counter(species_per_segment)
    ax.bar(counts.keys(), counts.values(), color="#9b59b6", edgecolor="none")
    ax.set_title("Species Count Per Soundscape Segment")
    ax.set_xlabel("Number of species in segment")
    ax.set_ylabel("Segment count")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "soundscape_density.png"), dpi=150)
    plt.close()
    print("Saved soundscape_density.png")

# ============================================================
# STEP 12 — RARE SPECIES DETAIL TABLE
# ============================================================

print("\n" + "=" * 60)
print("STEP 12 — RARE SPECIES DETAIL TABLE")
print("=" * 60)

rare_df = label_counts[label_counts < 20].reset_index()
rare_df.columns = ["primary_label", "count"]

# Merge with taxonomy for readable names
if "common_name" in taxonomy_df.columns:
    rare_df = rare_df.merge(
        taxonomy_df[["primary_label", "common_name", "class_name"]],
        on="primary_label",
        how="left"
    )

rare_df["is_zero_shot"] = rare_df["primary_label"].isin(zero_shot)
rare_df = rare_df.sort_values("count")

print(rare_df.to_string(index=False))
rare_df.to_csv(os.path.join(REPORTS_DIR, "rare_species.csv"), index=False)
print(f"\nSaved rare_species.csv ({len(rare_df)} species)")

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)

summary = f"""
========================
BIRDCLEF 2026 EDA REPORT
========================

--- DATASET ---
Total recordings          : {len(train_df)}
Unique species (train)    : {train_df['primary_label'].nunique()}
Unique species (taxonomy) : {taxonomy_df['primary_label'].nunique()}
Zero-shot species         : {len(zero_shot)}
Duplicate rows            : {train_df.duplicated().sum()}

--- CLASS IMBALANCE ---
Species with <10 samples  : {rare_10}
Species with <20 samples  : {rare_20}
Species with <50 samples  : {rare_50}

--- AUDIO ---
Average duration          : {dur_values.mean():.2f} sec
Median duration           : {np.median(dur_values):.2f} sec
Shortest clip             : {dur_values.min():.4f} sec
Longest clip              : {dur_values.max():.2f} sec
Corrupted files           : {len(corrupted_files)}
Files < 1 sec             : {len(very_short_files)}   (discard)
Files 1–2 sec             : {len(short_files)}   (repeat-pad)
Files > 600 sec           : {len(long_files)}   (suspicious)

--- SECONDARY LABELS ---
Recordings with sec labels: {secondary_exists} ({100*secondary_exists/len(train_df):.1f}%)

--- SOUNDSCAPE ---
Avg species per segment   : {species_per_segment.mean():.2f}
Max species in segment    : {int(species_per_segment.max())}

--- TENSOR ---
Target shape              : (3, {N_MELS}, {EXPECTED_COLS})
Shape verified            : {'YES' if shape_ok else 'NO — FIX BEFORE PHASE 2'}
"""

print(summary)

with open(os.path.join(REPORTS_DIR, "eda_summary.txt"), "w") as f:
    f.write(summary)

print("EDA COMPLETED SUCCESSFULLY.")
print(f"Plots   → {PLOTS_DIR}")
print(f"Reports → {REPORTS_DIR}")