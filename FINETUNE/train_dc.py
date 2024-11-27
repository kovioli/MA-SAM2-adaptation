import datetime
import os
from pathlib import Path
import random
from typing import Tuple, Dict, Any
import pdb

import numpy as np
import torch
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from monai.losses import GeneralizedDiceLoss

from _3D_HEAD.model_3D import HeadModel

from dataset_mrc import MRCDataset
from model import SAM2_finetune
from metrics import eval_seg
from config import (
    DEVICE,
    THRESHOLD,
    EPOCH,
    LR,
    BS,
    TRAIN_ID,
    VAL_DS_CONF,
    LOG_EVERY_STEP,
    MODEL_DICT,
    NR_SLICES,
    PROMPT_GRID,
    MIN_DELTA,
    PATIENCE,
    MODEL_TYPE,
)


class TrainingPipeline:
    def __init__(self, timestamp: str):
        self.timestamp = timestamp
        self.loss_fn = GeneralizedDiceLoss(sigmoid=True, reduction="mean")
        self.best_metrics = {"loss": float("inf"), "iou": 1, "dice": 1}
        self.epochs_without_improvement = 0

        # Setup paths
        base_path = Path("...")
        self.model_save_dir = base_path / "checkpoints" / timestamp
        self.model_save_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(base_path / "logs" / timestamp))

    def setup_model(self) -> Tuple[torch.nn.Module, torch.optim.Optimizer, GradScaler]:
        """Initialize model, optimizer and scaler."""
        # model = SAM2_finetune(
        #     model_cfg=MODEL_DICT[MODEL_TYPE]["config"],
        #     ckpt_path=MODEL_DICT[MODEL_TYPE]["ckpt"],
        #     device=DEVICE,
        #     use_point_grid=PROMPT_GRID,
        # )
        model = HeadModel(model_type=MODEL_TYPE, device=DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        scaler = GradScaler()
        return model, optimizer, scaler

    def setup_dataloaders(self, p: int) -> Tuple[DataLoader, DataLoader]:
        """Create training and validation dataloaders."""
        train_ds = MRCDataset(
            main_folder=".../EMPIAR_clean",
            ds_id=TRAIN_ID,
            s=NR_SLICES,
            p=p,
            device=DEVICE,
        )
        val_ds = MRCDataset(
            main_folder=".../EMPIAR_clean",
            ds_id=VAL_DS_CONF["ds_id"],
            s=VAL_DS_CONF["s"],
            p=VAL_DS_CONF["p"],
            device=DEVICE,
        )
        return (
            DataLoader(train_ds, batch_size=BS, shuffle=True),
            DataLoader(val_ds, batch_size=BS, shuffle=False),
        )

    @torch.no_grad()
    def evaluate(
        self, model: torch.nn.Module, dataloader: DataLoader, epoch: int
    ) -> Tuple[float, float, float]:
        """Evaluate model on validation set."""
        model.eval()
        metrics = {"loss": [], "iou": [], "dice": []}

        for image, label in dataloader:
            image = image.to(DEVICE)
            label = label.to(DEVICE)

            with autocast():
                pred, _ = model(image)
                loss = self.loss_fn(pred, label) * 100

            metrics["loss"].append(loss.item())
            iou, dice = eval_seg(pred, label, THRESHOLD)
            metrics["iou"].append(iou)
            metrics["dice"].append(dice)

        # Calculate means
        means = {k: np.mean(v) for k, v in metrics.items()}

        # Log to tensorboard
        for metric, value in means.items():
            self.writer.add_scalar(f"Validation/{metric.capitalize()}", value, epoch)

        print(
            f"| epoch {epoch:3d} | val loss {means['loss']:5.2f} | "
            f"iou {means['iou']:3.2f} | dice {means['dice']:3.2f}"
        )
        return means["loss"], means["iou"], means["dice"]

    def train_epoch(
        self,
        model: torch.nn.Module,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scaler: GradScaler,
        epoch: int,
        step: int,
    ) -> int:
        """Train for one epoch."""
        model.train()
        metrics = {"loss": [], "iou": [], "dice": []}

        for image, label in dataloader:
            image = image.to(DEVICE)
            label = label.to(DEVICE)

            optimizer.zero_grad()

            with autocast():
                pred, _ = model(image)
                loss = self.loss_fn(pred, label) * 100

            metrics["loss"].append(loss.item())
            iou, dice = eval_seg(pred, label, THRESHOLD)
            metrics["iou"].append(iou)
            metrics["dice"].append(dice)

            scaler.scale(loss).backward(retain_graph=True)
            scaler.step(optimizer)
            scaler.update()

            if step % LOG_EVERY_STEP == 0:
                recent_metrics = {
                    k: np.mean(v[-LOG_EVERY_STEP:]) for k, v in metrics.items()
                }
                for metric, value in recent_metrics.items():
                    self.writer.add_scalar(
                        f"TrainStep/{metric.capitalize()}", value, step
                    )

            step += 1

        # Calculate and log epoch metrics
        means = {k: np.mean(v) for k, v in metrics.items()}
        for metric, value in means.items():
            self.writer.add_scalar(f"Train/{metric.capitalize()}", value, epoch)

        print(
            f"| epoch {epoch:3d} | train loss {means['loss']:5.2f} | "
            f"iou {means['iou']:3.2f} | dice {means['dice']:3.2f}"
        )

        return step

    def stringify_p(self, p):
        if isinstance(p, int) or p.is_integer():
            return str(int(p))
        return str(float(p)).replace(".", "_")

    def log_run(self, p, r: int, best_iou: float, best_dice: float):
        """Log run configuration."""
        log_string = f"{self.timestamp},{MODEL_TYPE},{TRAIN_ID},s{NR_SLICES},p{self.stringify_p(p)},r{r},best_iou={best_iou:.4f},best_dice={best_dice:.4f}"
        log_path = Path("...") / f"log_s{NR_SLICES}.csv"
        print("LOGS:", log_string)
        with open(log_path, "a") as f:
            f.write(f"{log_string}\n")

    def train(self, p: int):
        """Main training loop."""

        # Setup
        model, optimizer, scaler = self.setup_model()
        train_dataloader, val_dataloader = self.setup_dataloaders(p)

        print(
            f"Loader lengths: train: {len(train_dataloader)}, val: {len(val_dataloader)}"
        )

        step = 0
        for epoch in range(EPOCH):
            epoch_start = datetime.datetime.now()

            # Train and evaluate
            step = self.train_epoch(
                model, train_dataloader, optimizer, scaler, epoch, step
            )
            val_loss, iou, dice = self.evaluate(model, val_dataloader, epoch)

            # Log timing
            epoch_duration = datetime.datetime.now() - epoch_start
            print(f"Epoch {epoch} duration: {epoch_duration}")

            # Check for improvement
            if val_loss < self.best_metrics["loss"] - MIN_DELTA * 100:
                self.best_metrics = {"loss": val_loss, "iou": iou, "dice": dice}
                self.epochs_without_improvement = 0
                print(
                    f"New best epoch {epoch}! "
                    f"Val loss: {val_loss:.4f}, IOU: {iou:.4f}, Dice: {dice:.4f}"
                )
                torch.save(model.state_dict(), self.model_save_dir / "best_model.pth")
            else:
                self.epochs_without_improvement += 1
                print(
                    f"Epoch {epoch} not best. "
                    f"Epochs without improvement: {self.epochs_without_improvement}"
                )

            # Early stopping check
            if self.epochs_without_improvement >= PATIENCE:
                print(f"Early stopping after {PATIENCE} epochs without improvement")
                break

        self.writer.close()
        print(
            f"Training finished! Best metrics:\n"
            f"Loss: {self.best_metrics['loss']:.4f}\n"
            f"IOU: {self.best_metrics['iou']:.4f}\n"
            f"Dice: {self.best_metrics['dice']:.4f}"
        )
        return self.best_metrics["iou"], self.best_metrics["dice"]


def main():
    p = 8
    while True:
        print("RUNNING FOR P:", p)
        for r in range(5):  # 5 runs
            print("RUNNING FOR R:", r)
            timestamp = datetime.datetime.now().strftime("%d%m%Y_%H:%M:%S")
            pipeline = TrainingPipeline(timestamp)
            best_iou, best_dice = pipeline.train(p)
            pipeline.log_run(p, r, best_iou, best_dice)
        p += 4


if __name__ == "__main__":
    main()
