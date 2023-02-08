import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import kornia as K
import pytorch_lightning as pl
import torchvision
from torch.utils.data import random_split, DataLoader
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torchvision.utils import make_grid
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning import loggers as pl_loggers
from torchvision.utils import save_image
import torchvision.transforms as T
import cv2
from scipy.spatial.distance import directed_hausdorff
from AImed.training import losses
from neptune.new.types import File
from AImed.model.dim2.unet import *
from AImed.model.dim2.utnetv2 import *

class model_selection():
    def __init__(self, flag = "UTNetV2"):
        self.flag = flag

    def select(self):
        if self.flag == "UTNetV2":
            return UTNetV2(1, 4)
        elif self.flag == "Unet":
            return UNet(1, 4)
        else:
            print("Please select valid model")
            exit()

def get_original_size(preds, label, size):
    # scale back to original size
    preds = torch.tensor(
            cv2.resize(
                preds.squeeze().cpu().numpy(),
                dsize=(size[1].numpy()[0], size[0].numpy()[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        ).float().unsqueeze(0)

    if label != None:
    
        label = (torch.tensor(cv2.resize(
                label.squeeze().cpu().numpy(),
                dsize=(size[1].numpy()[0], size[0].numpy()[0]),
                interpolation=cv2.INTER_NEAREST)).int().unsqueeze(0))

        return preds, label
    else:
        return preds

def compute_HF(label, preds, c):
    u = torch.argwhere(label.squeeze() == c)#.cpu().numpy()
    v = torch.argwhere(preds.squeeze() == c)#.cpu().numpy()
    HF = max(directed_hausdorff(u, v)[0], directed_hausdorff(v, u)[0])
    ##### camus dataset-> 0.145x0.154mm^2
    HF = HF*0.134
    return HF

def log_image(self, label, preds):
    new = torch.cat((label/self.classes, preds.unsqueeze(0)/ self.classes), dim=0)
    vis = torchvision.utils.make_grid(new, nrow=self.batch_size, padding=5)
    out_np = K.utils.tensor_to_image(vis)
    self.logger.experiment["validation/segmentations_pairs"].log(File.as_image(out_np))

def multi_class_dice_score(img, labels, class_labels=[1,2,3]):
    """ Given an image and a label compute the dice score over
    multiple class volumes. You can specify which classes dice
    should be computed for. Don't use zero because it's the background."""

    total_volume = 0.
    total_intersect_volume = 0.
    
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