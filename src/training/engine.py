import numpy as np
import torch
from tqdm import tqdm

from src.configs.config import *


# ============================================================
# TRAIN FUNCTION
# ============================================================

def train_fn(
    loader,
    model,
    optimizer,
    criterion
):

    model.train()

    total_loss = 0

    for batch in tqdm(loader):

        images = batch["image"].to(DEVICE)

        targets = batch["target"].to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            targets
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


# ============================================================
# VALID FUNCTION
# ============================================================

def valid_fn(
    loader,
    model,
    criterion
):

    model.eval()

    total_loss = 0

    predictions = []
    targets_all = []

    with torch.no_grad():

        for batch in tqdm(loader):

            images = batch["image"].to(DEVICE)

            targets = batch["target"].to(DEVICE)

            outputs = model(images)

            loss = criterion(
                outputs,
                targets
            )

            total_loss += loss.item()

            predictions.append(
                torch.sigmoid(outputs).cpu().numpy()
            )

            targets_all.append(
                targets.cpu().numpy()
            )

    predictions = np.concatenate(predictions)

    targets_all = np.concatenate(targets_all)

    return (
        total_loss / len(loader),
        predictions,
        targets_all
    )