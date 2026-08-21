import os
import pickle
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.amp import GradScaler, autocast
from torchmetrics.classification import Accuracy
from tqdm import tqdm

from src.model import EfficientCNN
from src.eia_ckd import HierarchicalConsistencyModel
from src.dataset import get_dataloaders


# ============================================================
# Reproducibility
# ============================================================

SEED = 5


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ============================================================
# Average Meter
# ============================================================

class AverageMeter:

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# ============================================================
# Training
# ============================================================

def train(
    trainer,
    train_loader,
    valid_loader,
    device,
    num_classes=5,
    num_epochs=50,
    t_max=80,
    learning_rate=3e-4,
    weight_decay=1e-5,
    temperature=3.0,
):

    os.makedirs(output_dir, exist_ok=True)

    optimizer = optim.AdamW(
        trainer.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=t_max,
    )

    ce_loss = nn.CrossEntropyLoss(
        label_smoothing=0.1
    )

    scaler = GradScaler(
        "cuda",
        enabled=(device.type == "cuda")
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    best_acc = 0.0

    # ========================================================
    # Epoch Loop
    # ========================================================

    for epoch in range(1, num_epochs + 1):

        # ----------------------------------------------------
        # Curriculum coefficients
        # ----------------------------------------------------

        beta_supervised = max(
            0.6,
            1.0 - (epoch / 80)
        )

        alpha_kl = min(
            1.0,
            epoch / 20
        )

        trainer.train()

        train_loss_meter = AverageMeter()

        train_acc_meter = Accuracy(
            task="multiclass",
            num_classes=num_classes
        ).to(device)

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch:02d}/{num_epochs}"
        )

        # ====================================================
        # Training batches
        # ====================================================

        for inputs, targets in pbar:

            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            with autocast(
                device_type=device.type,
                enabled=(device.type == "cuda")
            ):

                student_logits, consensus_logits = trainer(
                    inputs
                )

                # --------------------------------------------
                # Supervised Cross-Entropy Loss
                # --------------------------------------------

                loss_ce = ce_loss(
                    student_logits,
                    targets
                )

                # --------------------------------------------
                # Consistency KL Loss
                # --------------------------------------------

                loss_kl = F.kl_div(
                    F.log_softmax(
                        student_logits / temperature,
                        dim=1
                    ),
                    F.softmax(
                        consensus_logits.detach() / temperature,
                        dim=1
                    ),
                    reduction="batchmean"
                )

                # --------------------------------------------
                # Curriculum-based Loss
                # --------------------------------------------

                total_loss = (
                    beta_supervised * loss_ce
                    +
                    (1 - beta_supervised)
                    * (alpha_kl * loss_kl)
                )

            scaler.scale(total_loss).backward()

            scaler.step(optimizer)

            scaler.update()

            # --------------------------------------------
            # Metrics
            # --------------------------------------------

            train_loss_meter.update(
                total_loss.item(),
                inputs.size(0)
            )

            train_acc_meter.update(
                student_logits,
                targets
            )

            pbar.set_postfix({
                "Loss": f"{train_loss_meter.avg:.4f}",
                "Acc": (
                    f"{train_acc_meter.compute().item() * 100:.2f}%"
                ),
                "beta": f"{beta_supervised:.2f}",
                "alpha": f"{alpha_kl:.2f}",
            })

        scheduler.step()

        # ====================================================
        # Validation
        # ====================================================

        trainer.eval()

        val_loss_meter = AverageMeter()

        val_acc_meter = Accuracy(
            task="multiclass",
            num_classes=num_classes
        ).to(device)

        with torch.no_grad():

            for x, y in valid_loader:

                x = x.to(device)
                y = y.to(device)

                with autocast(
                    device_type=device.type,
                    enabled=(device.type == "cuda")
                ):

                    outputs = trainer.model(x)

                    val_loss = ce_loss(
                        outputs,
                        y
                    )

                val_loss_meter.update(
                    val_loss.item(),
                    x.size(0)
                )

                val_acc_meter.update(
                    outputs,
                    y
                )

        # ====================================================
        # Epoch statistics
        # ====================================================

        train_acc = train_acc_meter.compute().item()
        val_acc = val_acc_meter.compute().item()

        history["train_loss"].append(
            train_loss_meter.avg
        )

        history["val_loss"].append(
            val_loss_meter.avg
        )

        history["train_acc"].append(
            train_acc
        )

        history["val_acc"].append(
            val_acc
        )

        print(
            f"Epoch {epoch:02d} | "
            f"Train Acc: {train_acc * 100:.2f}% | "
            f"Val Acc: {val_acc * 100:.2f}% | "
            f"Val Loss: {val_loss_meter.avg:.4f}"
        )

        # ====================================================
        # Save Best Model
        # ====================================================

        if val_acc > best_acc:

            best_acc = val_acc

            torch.save(
                trainer.state_dict(),
                os.path.join(
                    output_dir,
                    "best_model.pth"
                )
            )

            print(
                f"Best model updated: "
                f"{best_acc * 100:.2f}%"
            )

        # ====================================================
