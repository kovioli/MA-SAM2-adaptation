# %%
import os
import time
import torch
from monai.losses import GeneralizedDiceLoss
from dataset import MRCDataset
from torch.utils.tensorboard import SummaryWriter
from _3D_HEAD.model_3D import HeadFinetuneModel
import datetime
import matplotlib.pyplot as plt
import numpy as np
from metrics import eval_seg
from torch.cuda.amp import autocast, GradScaler

from config import (
    DEVICE,
    EPOCH,
    LR,
    BS,
    DATA_DIR,
    SAVE_DIR,
    PREDICTION_CLASS,
    TRAIN_ID,
    VAL_ID,
    LOG_EVERY_STEP,
    PATIENCE,
    MIN_DELTA,
    THRESHOLD,
    MODEL_TYPE,
    MODEL_DICT,
)

lossfunc = GeneralizedDiceLoss(sigmoid=True, reduction="mean")


def evaluate(model, val_dataloader):
    model.eval()
    val_loss = []
    iou_list = []
    dice_list = []
    print("Val loader length: ", len(val_dataloader))
    with torch.no_grad():
        for image, label in val_dataloader:
            image = image.to(device=DEVICE)
            label = label.to(device=DEVICE)

            with autocast():
                pred, _ = model(image)
                loss = lossfunc(pred, label) * 100

            val_loss.append(loss.item())
            iou, dice = eval_seg(pred, label, THRESHOLD)
            iou_list.append(iou)
            dice_list.append(dice)

        loss_mean = np.average(val_loss)
        iou_mean = np.average(iou_list)
        dice_mean = np.average(dice_list)
        writer.add_scalar("Validation/Loss", loss_mean, epoch)
        writer.add_scalar("Validation/IoU", iou_mean, epoch)
        writer.add_scalar("Validation/Dice", dice_mean, epoch)

    print(
        f"| epoch {epoch:3d} | "
        f"val loss {loss_mean:5.2f} | "
        f"iou {iou_mean:3.2f}  | "
        f"dice {dice_mean:3.2f}"
    )
    return loss_mean, iou_mean, dice_mean


def evaluate_stepwise(model, val_dataloader, step):
    model.eval()
    val_loss = []
    iou_list = []
    dice_list = []
    with torch.no_grad():
        for image, label in val_dataloader:
            image = image.to(device=DEVICE)
            label = label.to(device=DEVICE)

            with autocast():
                pred, _ = model(image)
                loss = lossfunc(pred, label) * 100

            val_loss.append(loss.item())
            iou, dice = eval_seg(pred, label, THRESHOLD)
            iou_list.append(iou)
            dice_list.append(dice)

            loss_mean = np.average(val_loss)
            iou_mean = np.average(iou_list)
            dice_mean = np.average(dice_list)
            writer.add_scalar("ValStep/Loss", loss_mean, step)
            writer.add_scalar("ValStep/IoU", iou_mean, step)
            writer.add_scalar("ValStep/Dice", dice_mean, step)
    model.train()


def train(model, train_dataloader, test_dataloader, epoch, step):
    model.train()
    train_loss = []
    iou_list = []
    dice_list = []

    for image, label in train_dataloader:
        image = image.to(device=DEVICE)
        label = label.to(device=DEVICE)
        optimizer.zero_grad()

        with autocast():
            pred, _ = model(image)
            loss = lossfunc(pred, label) * 100

        train_loss.append(loss.item())

        scaler.scale(loss).backward(retain_graph=True)
        scaler.step(optimizer)
        scaler.update()

        iou, dice = eval_seg(pred, label, THRESHOLD)
        iou_list.append(iou)
        dice_list.append(dice)

        if step % LOG_EVERY_STEP == 0:
            loss_mean = np.average(train_loss[-LOG_EVERY_STEP:])
            iou_mean = np.average(iou_list[-LOG_EVERY_STEP:])
            dice_mean = np.average(dice_list[-LOG_EVERY_STEP:])
            writer.add_scalar("TrainStep/Loss", loss_mean, step)
            writer.add_scalar("TrainStep/IoU", iou_mean, step)
            writer.add_scalar("TrainStep/Dice", dice_mean, step)
            evaluate_stepwise(model, test_dataloader, step)

        step += 1

    loss_mean = np.average(train_loss)
    iou_mean = np.average(iou_list)
    dice_mean = np.average(dice_list)
    writer.add_scalar("Train/Loss", loss_mean, epoch)
    writer.add_scalar("Train/IoU", iou_mean, epoch)
    writer.add_scalar("Train/Dice", dice_mean, epoch)

    print(
        f"| epoch {epoch:3d} | "
        f"train loss {loss_mean:5.2f} | "
        f"iou {iou_mean:3.2f}  | "
        f"dice {dice_mean:3.2f}"
    )
    return step


if __name__ == "__main__":
    best_loss = float("inf")
    best_iou = 1
    best_dice = 1
    epochs_wo_improvement = 0

    model = HeadFinetuneModel(
        model_type=MODEL_TYPE,
        device=DEVICE,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scaler = GradScaler()

    timestamp_str = datetime.datetime.now().strftime("%d%m%Y_%H:%M:%S")
    print("TIMESTAMP: ", timestamp_str)
    model_save_dir = os.path.join(SAVE_DIR, "checkpoints", timestamp_str)
    os.makedirs(model_save_dir, exist_ok=True)

    # os.makedirs(os.path.dirname(model_save_dir), exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(SAVE_DIR, "logs", timestamp_str))

    train_ds = MRCDataset(
        data_folder=DATA_DIR,
        pred_class=PREDICTION_CLASS,
        ds_id=TRAIN_ID,
        device=DEVICE,
    )
    val_ds = MRCDataset(
        data_folder=DATA_DIR,
        pred_class=PREDICTION_CLASS,
        ds_id=VAL_ID,
        device=DEVICE,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_ds, batch_size=BS, shuffle=True
    )
    val_dataloader = torch.utils.data.DataLoader(val_ds, batch_size=BS, shuffle=False)

    print(f"Loader lengths: train: {len(train_dataloader)}, val: {len(val_dataloader)}")

    step = 0

    for epoch in range(EPOCH):
        epoch_start_time = time.time()
        step = train(model, train_dataloader, val_dataloader, epoch, step)
        elapsed_train = time.time() - epoch_start_time
        val_loss, iou, dice = evaluate(model, val_dataloader)
        elapsed_eval = time.time() - epoch_start_time - elapsed_train
        print(
            f"Time taken for epoch {epoch}: train: {int(elapsed_train)}s; eval: {int(elapsed_eval)}s"
        )

        # Check if this epoch is the best
        if (
            val_loss < best_loss - MIN_DELTA * 100
        ):  # loss is multiplied by 100 during evaluate()
            best_loss = val_loss
            best_iou = iou
            best_dice = dice
            epochs_wo_improvement = 0
            print(
                f"New best epoch! --> {epoch} with validation loss: {val_loss:.4f}, IOU: {iou:.4f}, Dice: {dice:.4f}"
            )
            torch.save(
                model.state_dict(), os.path.join(model_save_dir, f"best_model.pth")
            )
        else:
            epochs_wo_improvement += 1
            print(
                f"Epoch {epoch} was not the best. Epochs without improvement: {epochs_wo_improvement}"
            )

        # Early stopping check
        if epochs_wo_improvement >= PATIENCE:
            print(
                f"Early stopping triggered after {PATIENCE} epochs without improvement"
            )
            break
    writer.close()
    print("Training finished!")
    print(
        f"Best validation loss: {best_loss:.4f}, Best IOU: {best_iou:.4f}, Best Dice: {best_dice:.4f}"
    )
