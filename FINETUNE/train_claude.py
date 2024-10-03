import os
import time
import torch
import numpy as np
import datetime
from torch.utils.tensorboard import SummaryWriter
from monai.losses import GeneralizedDiceLoss
from shrec_dataset import create_multi_ds
from model import SAM2_finetune
from metrics import eval_seg

from config import (
    DEVICE, EPOCH, LR, BS, MODEL_TYPE, LOG_EVERY_STEP, PATIENCE,
    MIN_DELTA, THRESHOLD, TRAIN_IDs, VAL_IDs, MODEL_DICT
)

class Logger:
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir=log_dir)

    def log_scalar(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)

    def close(self):
        self.writer.close()

def create_model_and_optimizer():
    model = SAM2_finetune(
        model_cfg=MODEL_DICT[MODEL_TYPE]['config'],
        ckpt_path=MODEL_DICT[MODEL_TYPE]['ckpt'],
        device=DEVICE,
        use_point_grid=True
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    return model, optimizer

def evaluate(model, dataloader, logger, epoch, prefix='Validation'):
    model.eval()
    loss_func = GeneralizedDiceLoss(sigmoid=True, reduction='mean')
    losses, ious, dices = [], [], []

    with torch.no_grad():
        for image, label in dataloader:
            image, label = image.to(DEVICE), label.to(DEVICE)
            pred = model(image)
            loss = loss_func(pred, label) * 100
            iou, dice = eval_seg(pred, label, THRESHOLD)
            
            losses.append(loss.item())
            ious.append(iou)
            dices.append(dice)

    avg_loss, avg_iou, avg_dice = map(np.mean, [losses, ious, dices])
    logger.log_scalar(f'{prefix}/Loss', avg_loss, epoch)
    logger.log_scalar(f'{prefix}/IoU', avg_iou, epoch)
    logger.log_scalar(f'{prefix}/Dice', avg_dice, epoch)

    print(f"| epoch {epoch:3d} | {prefix.lower()} loss {avg_loss:5.2f} | iou {avg_iou:3.2f} | dice {avg_dice:3.2f}")
    model.train()
    return avg_loss, avg_iou, avg_dice

def train_epoch(model, train_dataloader, val_dataloader, optimizer, logger, epoch, step):
    model.train()
    loss_func = GeneralizedDiceLoss(sigmoid=True, reduction='mean')
    losses, ious, dices = [], [], []

    for image, label in train_dataloader:
        image, label = image.to(DEVICE), label.to(DEVICE)
        optimizer.zero_grad()
        pred = model(image)
        loss = loss_func(pred, label) * 100
        loss.backward()
        optimizer.step()

        iou, dice = eval_seg(pred, label, THRESHOLD)
        losses.append(loss.item())
        ious.append(iou)
        dices.append(dice)

        if step % LOG_EVERY_STEP == 0:
            log_metrics(logger, losses, ious, dices, step, 'TrainStep')
            evaluate(model, val_dataloader, logger, step, prefix='ValStep')

        step += 1

    log_metrics(logger, losses, ious, dices, epoch, 'Train')
    return step

def log_metrics(logger, losses, ious, dices, step, prefix):
    avg_loss, avg_iou, avg_dice = map(np.mean, [losses[-LOG_EVERY_STEP:], ious[-LOG_EVERY_STEP:], dices[-LOG_EVERY_STEP:]])
    logger.log_scalar(f'{prefix}/Loss', avg_loss, step)
    logger.log_scalar(f'{prefix}/IoU', avg_iou, step)
    logger.log_scalar(f'{prefix}/Dice', avg_dice, step)

def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

def train(model, train_dataloader, val_dataloader, optimizer, logger, num_epochs):
    best_loss = float('inf')
    epochs_without_improvement = 0
    step = 0

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        step = train_epoch(model, train_dataloader, val_dataloader, optimizer, logger, epoch, step)
        train_time = time.time() - epoch_start_time

        val_loss, val_iou, val_dice = evaluate(model, val_dataloader, logger, epoch)
        eval_time = time.time() - epoch_start_time - train_time

        print(f"Time taken for epoch {epoch}: train: {int(train_time)}s; eval: {int(eval_time)}s")

        if val_loss < best_loss - MIN_DELTA * 100:
            best_loss, best_iou, best_dice = val_loss, val_iou, val_dice
            epochs_without_improvement = 0
            print(f"New best epoch! --> {epoch} with validation loss: {val_loss:.4f}, IOU: {val_iou:.4f}, Dice: {val_dice:.4f}")
            save_model(model, os.path.join(model_save_dir, 'best_model.pth'))
        else:
            epochs_without_improvement += 1
            print(f"Epoch {epoch} was not the best. Epochs without improvement: {epochs_without_improvement}")

        if epochs_without_improvement >= PATIENCE:
            print(f"Early stopping triggered after {PATIENCE} epochs without improvement")
            break

    return best_loss, best_iou, best_dice

if __name__ == "__main__":
    timestamp_str = datetime.datetime.now().strftime("%d%m%Y_%H:%M")
    model_save_dir = os.path.join('/media', 'hdd1', 'oliver', 'SAM2_SHREC_FINETUNE', 'checkpoints', timestamp_str)
    os.makedirs(model_save_dir, exist_ok=True)

    logger = Logger(os.path.join('/media', 'hdd1', 'oliver', 'SAM2_SHREC_FINETUNE', 'logs', timestamp_str))

    train_data, val_data = create_multi_ds(
        main_folder=os.path.join('/media', 'hdd1', 'oliver', 'SHREC'),
        train_DS_IDs=TRAIN_IDs,
        val_DS_IDs=VAL_IDs,
        device=DEVICE
    )
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=BS, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=BS, shuffle=False)
    print(f"Loader lengths: train: {len(train_dataloader)}, val: {len(val_dataloader)}")

    model, optimizer = create_model_and_optimizer()
    best_loss, best_iou, best_dice = train(model, train_dataloader, val_dataloader, optimizer, logger, EPOCH)

    logger.close()
    print("Training finished!")
    print(f"Best validation loss: {best_loss:.4f}, Best IOU: {best_iou:.4f}, Best Dice: {best_dice:.4f}")