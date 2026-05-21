# BirdCLEF 2026 — End-to-End Bird Sound Classification Pipeline

## Overview

This project is a complete end-to-end deep learning pipeline built for the BirdCLEF 2026 Competition.

Competition Dataset:
https://www.kaggle.com/competitions/birdclef-2026/data

The objective of BirdCLEF 2026 is to identify bird species from real-world environmental audio recordings containing:

* overlapping species
* background noise
* weather distortions
* distant bird calls
* complex soundscapes

This project evolved from a simple baseline classifier into a competition-grade ecological audio intelligence system using:

* spectrogram-based deep learning
* soundscape adaptation
* pseudo labeling
* ensemble inference
* threshold optimization
* test-time augmentation (TTA)

---

# Dataset

Dataset Source:
https://www.kaggle.com/competitions/birdclef-2026/data

Dataset contains:

* training bird audio clips
* long soundscape recordings
* taxonomy metadata
* hidden test soundscapes

Audio Format:

* `.ogg`

Sampling Rate:

* 32 kHz

Total Classes:

* 206 bird species

---

# Project Goals

The main goals of this project were:

* Build a scalable BirdCLEF training pipeline
* Learn bird acoustic representations from spectrograms
* Improve robustness on noisy soundscapes
* Reduce overfitting using augmentation
* Improve generalization using pseudo labels
* Create a complete inference + submission system

---

# Tech Stack

## Deep Learning

* PyTorch
* timm

## Audio Processing

* librosa
* numpy

## Machine Learning

* scikit-learn
* pandas

## Training Utilities

* tqdm
* AMP (mixed precision ready)

---

# Project Pipeline

# PHASE 1 — Audio Processing & Spectrogram Pipeline

## Goal

Convert raw bird audio into learnable visual representations.

## Implemented

* audio loading
* resampling
* mel spectrogram generation
* log scaling
* 3-channel spectrogram tensors

## Approach

Audio recordings were transformed into Mel Spectrograms because spectrograms convert acoustic information into image-like structures that CNNs can learn effectively.

---

# PHASE 2 — Baseline EfficientNet Training

## Goal

Build the first stable bird classifier.

## Architecture

Spectrogram
→ EfficientNet-B0
→ Global Pooling
→ Dropout
→ Linear Classification Head
→ 206 species logits

## Why EfficientNet?

EfficientNet was selected because:

* lightweight architecture
* excellent image feature extraction
* proven BirdCLEF performance
* pretrained on ImageNet
* strong transfer learning capabilities

## Training Setup

### Loss Function

* BCEWithLogitsLoss

### Optimizer

* AdamW

### Scheduler

* CosineAnnealingLR

### Validation

* GroupKFold
* leakage-safe split

### Metric

* Macro ROC-AUC

---

# PHASE 3 — Stable End-to-End Training Pipeline

## Goal

Create a reproducible full training system.

## Implemented

* training loop
* validation loop
* checkpoint saving
* logging
* AUC tracking

## Outputs

* best model checkpoints
* validation metrics
* reproducible training pipeline

---

# PHASE 3.5 — Spectrogram Caching System

## Goal

Accelerate training dramatically.

## Problem

Real-time spectrogram generation on CPU was extremely slow.

## Solution

Implemented offline spectrogram caching.

All spectrograms were precomputed and stored as:

* `.npy tensors`

## Benefits

* major training speed improvement
* reduced CPU bottleneck
* faster experimentation

---

# PHASE 4 — Advanced Audio Augmentation

## Goal

Improve robustness and generalization.

## Implemented Augmentations

### SpecAugment

* time masking
* frequency masking

### Mixup

* spectrogram blending
* label blending

### Random Time Shift

* temporal robustness

### Random Gain

* volume robustness

## Why Augmentation?

Real BirdCLEF soundscapes contain:

* environmental noise
* partial bird calls
* distant species
* overlapping sounds

Augmentations simulated these conditions.

---

# PHASE 4.5 — Augmentation Stabilization

## Goal

Reduce over-augmentation.

## Changes

* reduced Mixup probability
* lighter SpecAugment masks
* controlled augmentation strength

## Result

More stable validation performance.

---

# PHASE 5 — Soundscape Adaptation

## Goal

Adapt model from clean clips to real-world soundscapes.

## Implemented

* soundscape window generation
* 5-second segmentation
* soundscape caching
* soundscape fine-tuning

## Why This Matters

Competition test data contains long noisy soundscapes instead of isolated clean clips.

Training directly on soundscape windows improved:

* ecological realism
* temporal robustness
* competition alignment

---

# PHASE 6 — Advanced Model Exploration

## Goal

Improve feature extraction capability.

## Experiments

* EfficientNet-B0
* PANN-inspired audio modeling
* soundscape fine-tuned architectures

## Observation

EfficientNet + strong training pipeline outperformed heavier CPU-limited experiments.

---

# PHASE 7 — Pseudo Labeling

## Goal

Leverage unlabeled soundscape data.

## Pipeline

Trained Model
→ Predict high-confidence labels
→ Generate pseudo labels
→ Retrain on combined dataset

## Why This Works

Pseudo labeling allows:

* semi-supervised learning
* improved generalization
* stronger soundscape adaptation

## Result

Significant validation improvement.

---

# PHASE 8 — Ensemble Inference

## Goal

Improve prediction stability.

## Implemented

Ensemble averaging between:

* soundscape model
* pseudo-labeled model

## Benefit

Reduced variance and improved robustness.

---

# PHASE 8B — Threshold Optimization

## Goal

Improve multi-label decision calibration.

## Problem

Different species require different confidence thresholds.

## Solution

Optimized per-class thresholds using validation predictions.

## Benefit

Better precision/recall balance.

---

# PHASE 8C — Test-Time Augmentation (TTA)

## Goal

Improve inference robustness.

## TTA Variants

* original spectrogram
* shifted spectrogram
* gain-adjusted spectrogram

## Final Prediction

Average of all augmented predictions.

## Benefit

* more stable predictions
* improved robustness
* better confidence calibration

---

# PHASE 9 — Full Submission Pipeline

## Goal

Create complete competition inference system.

## Final Pipeline

Raw Audio
→ Windowing
→ Mel Spectrogram
→ Ensemble Models
→ TTA
→ Threshold Optimization
→ submission.csv

## Implemented

* automatic soundscape windowing
* ensemble inference
* threshold application
* CSV submission generation

---

# Final System Features

## Implemented Features

* End-to-end BirdCLEF pipeline
* Spectrogram caching
* EfficientNet training
* Grouped validation
* Soundscape adaptation
* Pseudo labeling
* Ensemble inference
* Threshold optimization
* Test-time augmentation
* Submission generation

---

# Model Performance

## Best Validation AUC

0.88+ Macro ROC-AUC

using:

* soundscape fine-tuning
* pseudo labeling
* grouped validation
* ensemble inference

---

# Folder Structure

```bash
birdclef-2026/
│
├── checkpoints/
├── spectrogram_cache/
├── soundscape_cache/
├── src/
│   ├── configs/
│   ├── datasets/
│   ├── models/
│   ├── training/
│   └── utils/
│
├── train.py
├── train_soundscape.py
├── train_pseudo.py
├── generate_submission.py
├── ensemble_inference.py
├── tta_inference.py
└── optimize_thresholds.py
```

---

# Key Learnings

This project demonstrated:

* ecological audio modeling
* spectrogram-based deep learning
* semi-supervised learning
* competition-grade ML engineering
* large-scale audio inference pipelines

---

# Future Improvements

Potential future upgrades:

* multi-fold cross validation
* sliding window inference
* HTS-AT transformers
* BEATs audio transformers
* BirdNET encoders
* external bird datasets
* GPU acceleration

---

# Acknowledgements

* BirdCLEF organizers
* Kaggle community
* PyTorch ecosystem
* timm pretrained models

---

# Author

## Sahil Poply

B.Tech CSE Student
Machine Learning & Audio AI Enthusiast

Focused on:

* Deep Learning
* Audio Intelligence
* Real-World AI Systems
* Competition ML Pipelines
