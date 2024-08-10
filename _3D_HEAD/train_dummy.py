# from model import HeadModel
from model import HeadModel
from dataset import PNGDataset
from metrics import eval_seg
from datetime import datetime

from monai.losses import GeneralizedDiceLoss
import os
import time
import yaml
import torch
import torch.nn as nn
import numpy as np
EPOCHS = 5
DEVICE = 'cuda:1'
LR = 1e-3
THRESHOLD = (0.1, 0.3, 0.5, 0.7, 0.9)
lossfunc = GeneralizedDiceLoss(sigmoid=True, reduction='mean')
torch.autograd.set_detect_anomaly(True)

def train(model, train_dataloader, epoch, step):
    model.train()
    train_loss = []
    memory = None
    for image, label in train_dataloader:
        image = image.to(device=DEVICE)
        label = label.to(device=DEVICE)
        
        optimizer.zero_grad()
        pred = model(image)
        # pred, mem = model(image, None)
        # memory = tuple(m.detach() for m in mem)
        loss = lossfunc(pred, label) * 100
        loss.backward(retain_graph=True) # possibly retain_graph=True
        optimizer.step()
        train_loss.append(loss.item())
        iou, dice = eval_seg(pred, label, THRESHOLD)
        step += 1
        if step % 10 == 0:
            print(f"Epoch: {epoch}, Step: {step}, Loss: {loss.item()}, IoU: {iou}, Dice: {dice}")
    # return step, np.average(train_loss)
    return step
    
if __name__ == '__main__':
    timestamp_str = datetime.now().strftime("%d%m%Y_%H:%M")
    model_save_dir = os.path.join(
        '/oliver',
        'SAM2',
        'checkpoints',
        timestamp_str
    )
    os.makedirs(model_save_dir, exist_ok=True)
    model = HeadModel('tiny', DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)    
    train_dataset = PNGDataset(
        main_folder='/oliver/EMPIAR_png',
        DS_ID='TS_0001',
        device=DEVICE
    )
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False)
    
    step = 0
    for epoch in range(EPOCHS):
        epoch_start_time = time.time()
        step = train(model, train_dataloader, epoch, step)
        elpased_train = time.time() - epoch_start_time
        print(f"Epoch: {epoch}, Time: {elpased_train}")
        torch.save(model.state_dict(), os.path.join(model_save_dir, f"model_{epoch}.pt"))