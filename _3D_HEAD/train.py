import pdb
import re
import os
import time
import yaml
import torch
import torch.nn as nn
import numpy as np
# from dataset import create_train_val_datasets
from shrec_dataset import create_multi_ds
from torch.autograd import Function
from torchvision import transforms
from monai.losses import GeneralizedDiceLoss, DiceCELoss
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import datetime
from model_3D import HeadModel

from metrics import eval_seg

from config import (
    DEVICE,
    THRESHOLD,
    EPOCH,
    LR,
    BS,
    TRAIN_IDs,
    LOG_EVERY_STEP,
    TRAIN_RATIO,
    MIN_DELTA,
    PATIENCE,
    MODEL_TYPE,
    VAL_IDs
)    


lossfunc = GeneralizedDiceLoss(sigmoid=True, reduction='mean') # original (deepict), has been using for quite a while
# lossfunc = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')


def evaluate(model,val_dataloader):
    model.eval()
    val_loss = []
    iou_list = []
    dice_list = []
    print("Val loader length: ", len(val_dataloader))
    memory = None
    with torch.no_grad():
        for image, label in val_dataloader:
            image = image.to(device=DEVICE)
            label = label.to(device=DEVICE)
            
            # pred, mem = model(image, memory)
            # memory = tuple(m.detach() for m in mem)
            pred, _ = model(image)

            loss = lossfunc(pred,label) * 100
            val_loss.append(loss.item())
            iou,dice = eval_seg(pred, label, THRESHOLD)
            iou_list.append(iou)
            dice_list.append(dice)

        loss_mean = np.average(val_loss)
        iou_mean = np.average(iou_list)
        dice_mean = np.average(dice_list)
        writer.add_scalar('Validation/Loss', loss_mean, epoch)
        writer.add_scalar('Validation/IoU', iou_mean, epoch)
        writer.add_scalar('Validation/Dice', dice_mean, epoch)

    print(
        f"| epoch {epoch:3d} | "f"val loss {loss_mean:5.2f} | "f"iou {iou_mean:3.2f}  | "f"dice {dice_mean:3.2f}"
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
            
            pred = model(image)
            loss = lossfunc(pred, label) * 100
            val_loss.append(loss.item())
            iou, dice = eval_seg(pred, label, THRESHOLD)
            iou_list.append(iou)
            dice_list.append(dice)

            loss_mean = np.average(val_loss)
            iou_mean = np.average(iou_list)
            dice_mean = np.average(dice_list)
            writer.add_scalar('ValStep/Loss', loss_mean, step)
            writer.add_scalar('ValStep/IoU', iou_mean, step)
            writer.add_scalar('ValStep/Dice', dice_mean, step)
    model.train()

def train(model,train_dataloader, test_dataloader, epoch, step):
    model.train()
    train_loss = []
    iou_list = []
    dice_list = []
    memory = None

    for image, label in train_dataloader:
        image = image.to(device=DEVICE)
        label = label.to(device=DEVICE)
        optimizer.zero_grad()

        # pred, mem = model(image, memory)
        pred, _ = model(image)
        #memory = tuple(m.detach() for m in mem)
        loss = lossfunc(pred, label) * 100
        train_loss.append(loss.item())
        loss.requires_grad_(True)
        loss.backward(retain_graph=True)
        optimizer.step()
        # pdb.set_trace()
        # for name, param in model.unet.named_parameters():
        #     if param.requires_grad:
        #         print(f"Gradient for {name}: {param.grad.mean().item()}")
        # print("\n\n")
        
        iou, dice = eval_seg(pred, label, THRESHOLD)
        iou_list.append(iou)
        dice_list.append(dice)
        
        if step % LOG_EVERY_STEP == 0:
            loss_mean = np.average(train_loss[-LOG_EVERY_STEP:])
            iou_mean = np.average(iou_list[-LOG_EVERY_STEP:])
            dice_mean = np.average(dice_list[-LOG_EVERY_STEP:])
            writer.add_scalar('TrainStep/Loss', loss_mean, step)
            writer.add_scalar('TrainStep/IoU', iou_mean, step)
            writer.add_scalar('TrainStep/Dice', dice_mean, step)
            # evaluate_stepwise(model, test_dataloader, step)
        
        step += 1

    loss_mean = np.average(train_loss)
    iou_mean = np.average(iou_list)
    dice_mean = np.average(dice_list)
    writer.add_scalar('Train/Loss', loss_mean, epoch)
    writer.add_scalar('Train/IoU', iou_mean, epoch)
    writer.add_scalar('Train/Dice', dice_mean, epoch)

    print(
        f"| epoch {epoch:3d} | "f"train loss {loss_mean:5.2f} | "f"iou {iou_mean:3.2f}  | "f"dice {dice_mean:3.2f}"
    )
    return step


if __name__ == "__main__":
    
    model = HeadModel(MODEL_TYPE, DEVICE, in_chan=3)
    best_loss = float('inf')
    best_iou = 1
    best_dice = 1
    epochs_wo_improvement = 0

    timestamp_str = datetime.datetime.now().strftime("%d%m%Y_%H:%M")
    print("TIMESTAMP: ", timestamp_str)
    model_save_dir = os.path.join(
        '/media',
        'hdd1',
        'oliver',
        'SAM2_SHREC',
        'checkpoints',
        timestamp_str
    )
    os.makedirs(model_save_dir, exist_ok=True)

    #os.makedirs(os.path.dirname(model_save_dir), exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join('/media', 'hdd1', 'oliver', 'SAM2_SHREC', 'logs', timestamp_str))
    


    # train_data_PNG, val_data_PNG = create_train_val_datasets(
    #     main_folder=os.path.join('/media', 'hdd1', 'oliver', 'SHREC'),
    #     DS_ID=TRAIN_ID,
    #     device=DEVICE
    # )
    # train_data_PNG, val_data_PNG = create_multi_ds(
    #     main_folder=os.path.join('/media', 'hdd1', 'oliver', 'SHREC'),
    #     DS_IDs=TRAIN_IDs,
    #     device=DEVICE
    # )
    train_data_PNG, val_data_PNG = create_multi_ds(
        main_folder=os.path.join('/media', 'hdd1', 'oliver', 'SHREC'),
        train_DS_IDs=TRAIN_IDs,
        val_DS_IDs=VAL_IDs,
        device=DEVICE
    )

    train_dataloader = torch.utils.data.DataLoader(train_data_PNG, batch_size=BS, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_data_PNG, batch_size=BS, shuffle=False)  

    print(f"Loader lengths: train: {len(train_dataloader)}, val: {len(val_dataloader)}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)    

    step = 0

    for epoch in range(EPOCH):
        epoch_start_time = time.time()
        step = train(model, train_dataloader, val_dataloader, epoch, step)
        elapsed_train = time.time() - epoch_start_time
        val_loss, iou, dice = evaluate(model, val_dataloader)
        elapsed_eval = time.time() - epoch_start_time - elapsed_train
        print(f"Time taken for epoch {epoch}: train: {int(elapsed_train)}s; eval: {int(elapsed_eval)}s")

        # Check if this epoch is the best
        if val_loss < best_loss - MIN_DELTA * 100: # loss is multiplied by 100 during evaluate()
            best_loss = val_loss
            best_iou = iou
            best_dice = dice
            epochs_wo_improvement = 0
            print(f"New best epoch! --> {epoch} with validation loss: {val_loss:.4f}, IOU: {iou:.4f}, Dice: {dice:.4f}")
            torch.save(model.state_dict(), os.path.join(model_save_dir, f'best_model.pth'))
        else:
            epochs_wo_improvement += 1
            print(f"Epoch {epoch} was not the best. Epochs without improvement: {epochs_wo_improvement}")

        # Early stopping check
        if epochs_wo_improvement >= PATIENCE:
            print(f"Early stopping triggered after {PATIENCE} epochs without improvement")
            break
    writer.close()
    print("Training finished!")
    print(f"Best validation loss: {best_loss:.4f}, Best IOU: {best_iou:.4f}, Best Dice: {best_dice:.4f}")