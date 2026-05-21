import timm
import torch
import torch.nn as nn

from src.configs.config import *


class PANNsModel(nn.Module):

    def __init__(self):

        super().__init__()

        # ====================================================
        # AUDIO BACKBONE
        # ====================================================

        self.backbone = timm.create_model(
            "eca_nfnet_l0",
            pretrained=True,
            in_chans=3,
            num_classes=0
        )

        feature_dim = self.backbone.num_features

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

        features = self.backbone(
            x
        )

        # ====================================================
        # DROPOUT
        # ====================================================

        features = self.dropout(
            features
        )

        # ====================================================
        # LOGITS
        # ====================================================

        logits = self.classifier(
            features
        )

        return logits