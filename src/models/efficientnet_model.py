import timm
import torch
import torch.nn as nn

from src.configs.config import *

from src.models.attention import AttentionPooling


class BirdCLEFModel(nn.Module):

    def __init__(self):

        super().__init__()

        # ====================================================
        # BACKBONE
        # ====================================================

        self.backbone = timm.create_model(
            MODEL_NAME,
            pretrained=True,
            in_chans=3,
            num_classes=0
        )

        # ====================================================
        # FEATURE DIM
        # ====================================================

        feature_dim = self.backbone.num_features

        # ====================================================
        # ATTENTION POOLING
        # ====================================================

        self.attention_pool = AttentionPooling(
            feature_dim
        )

        # ====================================================
        # DROPOUT
        # ====================================================

        self.dropout = nn.Dropout(
            DROPOUT
        )

        # ====================================================
        # CLASSIFIER
        # ====================================================

        self.classifier = nn.Linear(
            feature_dim,
            NUM_CLASSES
        )

    def forward(
        self,
        x
    ):

        # ====================================================
        # FEATURES
        # ====================================================

        features = self.backbone.forward_features(
            x
        )

        # ====================================================
        # SHAPE
        # [B, C, H, W]
        # ====================================================

        B, C, H, W = features.shape

        # ====================================================
        # TEMPORAL POOL OVER FREQUENCY
        # ====================================================

        features = features.mean(
            dim=2
        )

        # ====================================================
        # [B, C, T]
        # → [B, T, C]
        # ====================================================

        features = features.permute(
            0,
            2,
            1
        )

        # ====================================================
        # ATTENTION POOLING
        # ====================================================

        pooled = self.attention_pool(
            features
        )

        # ====================================================
        # DROPOUT
        # ====================================================

        pooled = self.dropout(
            pooled
        )

        # ====================================================
        # CLASSIFICATION
        # ====================================================

        logits = self.classifier(
            pooled
        )

        return logits