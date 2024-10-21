import torch.nn as nn
import torch
from torch.autograd import Function
import numpy as np
############### DeePiCt ###############
class DiceCoefficient(nn.Module): # -> simple metric for evaluation
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    # the dice coefficient of two sets represented as vectors a, b ca be
    # computed as (2 *|a b| / (a^2 + b^2))
    def forward(self, prediction, target):
        intersection = (prediction * target).sum()
        denominator = (prediction * prediction).sum() + (target * target).sum()
        return 2 * intersection / denominator.clamp(min=self.eps)
    

class DiceCoefficientLoss(nn.Module): # -> loss function, NOT WORKING
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
        # the dice coefficient of two sets represented as vectors a, b can be
        # computed as (2 *|a b| / (a^2 + b^2))

    def forward(self, input, target):
        input.float()
        target.float()
        dice_loss = 0
        channels = input.shape[1]
        # print("channels = ", channels)
        # print("input.shape", input.shape)
        for channel in range(channels):
            channel_prediction = input[:, channel, ...].float()
            channel_target = target[:, channel, ...].float()
            intersection = (channel_prediction * channel_target).sum()
            denominator = (channel_prediction * channel_prediction).sum() + (
                    channel_target * channel_target).sum()
            dice_loss += (1 - 2 * intersection / denominator.clamp(
                min=self.eps))
        dice_loss /= channels  # normalize loss
        return dice_loss
############# END DeePiCt ##############


############### SAM-adap ###############
class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 0.0001
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps

        t = (2 * self.inter.float() + eps) / self.union.float()
        return t
    
############# END SAM-adap #############



def cal_iou(outputs: np.array, labels: np.array):
    SMOOTH = 1e-6
    intersection = (outputs & labels).sum((1, 2))
    union = (outputs | labels).sum((1, 2))

    iou = (intersection + SMOOTH) / (union + SMOOTH)

    return iou.mean()

def dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).to(device = input.device).zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])

    return s / (i + 1)

def eval_seg(pred, true_mask_p, threshold):

    eiou, edice = 0, 0
    for th in threshold:
        gt_vmask_p = (true_mask_p > th).float()
        vpred = (pred > th).float()
        vpred_cpu = vpred.cpu()
        disc_pred = vpred_cpu[:, 0, :, :].numpy().astype('int32')

        disc_mask = gt_vmask_p[:, 0, :, :].squeeze(1).cpu().numpy().astype('int32')

        '''iou for numpy'''
        eiou += cal_iou(disc_pred, disc_mask)

        '''dice for torch'''
        edice += DiceCoefficient()(vpred, gt_vmask_p).item()
        #edice += dice_coeff(vpred[:, 0, :, :], gt_vmask_p[:, 0, :, :]).item()

    return eiou / len(threshold), edice / len(threshold)