import torch
import torch.nn as nn


class AttentionPooling(nn.Module):

    def __init__(
        self,
        in_features
    ):

        super().__init__()

        self.attention = nn.Sequential(

            nn.Linear(
                in_features,
                128
            ),

            nn.Tanh(),

            nn.Linear(
                128,
                1
            )
        )

    def forward(
        self,
        x
    ):

        # ================================================
        # x shape:
        # [B, T, C]
        # ================================================

        attention_weights = self.attention(
            x
        )

        attention_weights = torch.softmax(
            attention_weights,
            dim=1
        )

        weighted = x * attention_weights

        pooled = weighted.sum(
            dim=1
        )

        return pooled