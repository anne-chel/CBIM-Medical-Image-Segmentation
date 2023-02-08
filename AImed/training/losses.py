import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb
import kornia as K

class CE_Loss(nn.Module):
    def __init__(self, weight = [0.5, 1, 1, 1]):
        super(CE_Loss, self).__init__()
        self.weight = weight
        self.f1 = nn.CrossEntropyLoss(weight=torch.tensor(self.weight))
        self.log_list = []

    def forward(self, preds, targets):
        loss = self.f1(preds, targets)
        self.log_list = [["CE_loss", loss]]
        return loss

class Dice_CE_Loss(nn.Module):
    def __init__(self, weight=[0.5, 1, 1, 1]):
        super(CELoss, self).__init__()
        self.weight = weight
        self.f1 = nn.CrossEntropyLoss(weight=torch.tensor(self.weight))
        self.f2 = DiceLoss()
        self.log_list = []

    def forward(self, preds, targets):
        ce_loss = self.f1(preds, targets)
        dice_loss = self.f2(preds, targets)
        loss = ce_loss + dice_loss
        self.log_list = [["CE_loss", ce_loss], ["D_loss", dice_loss]]
        return loss

class Dice_CE_Edge_Loss(nn.Module):
    def __init__(self, weight=[0.5, 1, 1, 1], ignore_index=-10, lamd = 0.2):
        super(CELoss, self).__init__()
        self.weight = weight
        self.ignore_index = ignore_index
        self.lamd = lamd
        self.f1 = nn.CrossEntropyLoss(weight=torch.tensor(self.weight))
        self.f2 = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
        self.f3 = losses.DiceLoss()
        self.log_list = []

    def forward(self, preds, targets):
        CE_loss = self.f1(preds, targets)

        # find the edge pixels for the segmentation classes
        x_sobel = K.filters.sobel(targets.unsqueeze(0) / 3)
        reverse = 1.0 - x_sobel

        # set non-edge pixels to -10 such that these will be ignored in the loss
        edges = torch.where(reverse > 0.9989, -10, targets.unsqueeze(0)).squeeze(1)

        reg_loss = self.criterion2(preds, edges)
        D_loss = self.criterion3(preds, targets.unsqueeze(0))
        loss = CE_loss + self.lamd * reg_loss + D_loss

        self.log_list = [["CE_loss", CE_loss], ["D_loss", D_loss], ["Edge_loss", reg_loss]]
        return loss

class DiceLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, size_average=True, reduce=True):
        super(DiceLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

        self.size_average = size_average
        self.reduce = reduce
        self.log_list = []

    def forward(self, preds, targets):
        N = preds.size(0)
        C = preds.size(1)

        P = F.softmax(preds, dim=1)
        smooth = torch.zeros(C, dtype=torch.float32).fill_(0.00001)

        class_mask = torch.zeros(preds.shape).to(preds.device)
        class_mask.scatter_(1, targets, 1.0)

        ones = torch.ones(preds.shape).to(preds.device)
        P_ = ones - P
        class_mask_ = ones - class_mask

        TP = P * class_mask
        FP = P * class_mask_
        FN = P_ * class_mask

        smooth = smooth.to(preds.device)
        self.alpha = FP.transpose(0, 1).reshape(C, -1).sum(dim=(1)) / (
            (
                FP.transpose(0, 1).reshape(C, -1).sum(dim=(1))
                + FN.transpose(0, 1).reshape(C, -1).sum(dim=(1))
            )
            + smooth
        )

        self.alpha = torch.clamp(self.alpha, min=0.2, max=0.8)
        # print('alpha:', self.alpha)
        self.beta = 1 - self.alpha
        num = torch.sum(TP.transpose(0, 1).reshape(C, -1), dim=(1)).float()
        den = (
            num
            + self.alpha * torch.sum(FP.transpose(0, 1).reshape(C, -1), dim=(1)).float()
            + self.beta * torch.sum(FN.transpose(0, 1).reshape(C, -1), dim=(1)).float()
        )

        dice = num / (den + smooth)

        if not self.reduce:
            loss = torch.ones(C).to(dice.device) - dice
            return loss

        loss = 1 - dice
        loss = loss.sum()

        if self.size_average:
            loss /= C

        self.log_list = [["D_loss", loss]]
        return loss


class FocalLoss(nn.Module):
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()

        if alpha is None:
            self.alpha = torch.ones(class_num)
        else:
            self.alpha = alpha

        self.gamma = gamma
        self.size_average = size_average

    def forward(self, preds, targets):
        N = preds.size(0)
        C = preds.size(1)

        targets = targets.unsqueeze(1)
        P = F.softmax(preds, dim=1)
        log_P = F.log_softmax(preds, dim=1)

        class_mask = torch.zeros(preds.shape).to(preds.device)
        class_mask.scatter_(1, targets, 1.0)

        if targets.size(1) == 1:
            # squeeze the chaneel for target
            targets = targets.squeeze(1)
        alpha = self.alpha[targets.data].to(preds.device)

        probs = (P * class_mask).sum(1)
        log_probs = (log_P * class_mask).sum(1)

        batch_loss = -alpha * (1 - probs).pow(self.gamma) * log_probs

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()

        self.log_list = [["Focal_loss", loss]]

        return loss

class loss_selection():
    def __init__(self, flag = "CE"):
        self.flag = flag

    def select(self):
        if self.flag == "CE":
            return CE_Loss()
        elif self.flag == "CE+EDGE+Dice":
            return Dice_CE_Edge_Loss()
        elif self.flag == "Dice":
            return DiceLoss()
        elif self.flag == "Focal":
            return FocalLoss(class_num=4, gamma=2)
        elif self.flag == "CE+Dice":
            return Dice_CE_Loss()
        else:
            print("Please select valid loss function")
            exit()