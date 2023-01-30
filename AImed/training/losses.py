import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb

def multi_class_dice_score(self, img, labels, class_labels=[1, 2, 3]):
    """Given an image and a label compute the dice score over
    multiple class volumes. You can specify which classes dice
    should be computed for. Don't use zero because it's the background."""

    total_volume = 0.0
    total_intersect_volume = 0.0

    outputs = []
    for label in class_labels:
        img_bool = img.flatten() == label
        labels_bool = labels.flatten() == label

        volume = sum(img_bool) + sum(labels_bool)
        intersect_volume = sum(img_bool & labels_bool)

        total_volume += volume
        total_intersect_volume += intersect_volume

        outputs.append(2 * intersect_volume / volume)

    return 2 * total_intersect_volume / total_volume, outputs

def multi_class_jaccard(self, img, labels, class_labels=[1, 2, 3]):
    """Jaccard metric defined for two sets as |A and B| / |A or B|"""

    total_union_volume = 0.0
    total_intersect_volume = 0.0

    outputs = []
    for label in class_labels:
        img_bool = img.flatten() == label
        labels_bool = labels.flatten() == label

        union_volume = sum(img_bool | labels_bool)
        intersect_volume = sum(img_bool & labels_bool)

        total_union_volume += union_volume
        total_intersect_volume += intersect_volume

        outputs.append(intersect_volume / union_volume)

    return total_intersect_volume / total_union_volume, outputs


class DiceLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, size_average=True, reduce=True):
        super(DiceLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

        self.size_average = size_average
        self.reduce = reduce

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

        return loss


if __name__ == "__main__":

    DL = DiceLoss()
    FL = FocalLoss(10)

    pred = torch.randn(2, 10, 128, 128)
    target = torch.zeros((2, 1, 128, 128)).long()

    dl_loss = DL(pred, target)
    fl_loss = FL(pred, target)

    print("2D:", dl_loss.item(), fl_loss.item())

    pred = torch.randn(2, 10, 64, 128, 128)
    target = torch.zeros(2, 1, 64, 128, 128).long()

    dl_loss = DL(pred, target)
    fl_loss = FL(pred, target)

    print("3D:", dl_loss.item(), fl_loss.item())
