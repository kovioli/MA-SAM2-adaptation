# %%
import os
import time
import torch
from monai.losses import GeneralizedDiceLoss
from shrec_dataset import MRCDataset, create_multi_ds
from torch.utils.tensorboard import SummaryWriter
from model import SAM2_finetune
import datetime
import matplotlib.pyplot as plt
import numpy as np
from metrics import eval_seg

from config import (
    DEVICE,
    EPOCH,
    LR,
    BS,
    MODEL_TYPE,
    LOG_EVERY_STEP,
    PATIENCE,
    MIN_DELTA,
    THRESHOLD,
    TRAIN_IDs,
    VAL_IDs,
    MODEL_DICT,
    PROMPT_GRID
)

lossfunc = GeneralizedDiceLoss(sigmoid=True, reduction='mean')

def evaluate(model,val_dataloader):
    model.eval()
    val_loss = []
    iou_list = []
    dice_list = []
    print("Val loader length: ", len(val_dataloader))
    with torch.no_grad():
        for image, label in val_dataloader:
            image = image.to(device=DEVICE)
            label = label.to(device=DEVICE)
            
            pred = model(image)

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

    for image, label in train_dataloader:
        image = image.to(device=DEVICE)
        label = label.to(device=DEVICE)
        optimizer.zero_grad()

        # pred, mem = model(image, memory)
        pred = model(image)
        loss = lossfunc(pred, label) * 100
        train_loss.append(loss.item())
        loss.requires_grad_(True)
        loss.backward(retain_graph=True)
        # loss.backward()
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
            evaluate_stepwise(model, test_dataloader, step)
        
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
    best_loss = float('inf')
    best_iou = 1
    best_dice = 1
    epochs_wo_improvement = 0
    
    model = SAM2_finetune(
        model_cfg=MODEL_DICT[MODEL_TYPE]['config'],
        ckpt_path=MODEL_DICT[MODEL_TYPE]['ckpt'],
        device=DEVICE,
        use_point_grid=PROMPT_GRID
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    timestamp_str = datetime.datetime.now().strftime("%d%m%Y_%H:%M")
    
    print("TIMESTAMP: ", timestamp_str)
    model_save_dir = os.path.join(
        '/media',
        'hdd1',
        'oliver',
        'SAM2_SHREC_FINETUNE',
        'checkpoints',
        timestamp_str
    )
    os.makedirs(model_save_dir, exist_ok=True)

    #os.makedirs(os.path.dirname(model_save_dir), exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join('/media', 'hdd1', 'oliver', 'SAM2_SHREC_FINETUNE', 'logs', timestamp_str))
    
    train_data, val_data = create_multi_ds(
        main_folder=os.path.join('/media', 'hdd1', 'oliver', 'SHREC'),
        train_DS_IDs=TRAIN_IDs,
        val_DS_IDs=VAL_IDs,
        device=DEVICE
    )
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=BS, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=BS, shuffle=False)
    print(f"Loader lengths: train: {len(train_dataloader)}, val: {len(val_dataloader)}")

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
        