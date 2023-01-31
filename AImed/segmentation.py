# -*- coding: utf-8 -*-

import os
import glob
import yaml
import argparse
import SimpleITK as sitk
from pprint import pprint
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch import tensor
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from AImed.model.dim2.unet import *
from AImed.model.dim2.utnetv2 import *
import kornia as K
import pytorch_lightning as pl
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torchvision.utils import make_grid
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning import loggers as pl_loggers
import numpy as np
import os
import torch
import numpy as np
from torch.utils.data import Dataset
import SimpleITK as sitk
import yaml
import random
from tqdm import tqdm
from neptune.new.types import File
#from training import losses
from pytorch_lightning.loggers import NeptuneLogger
import cv2
import skimage.exposure
import numpy as np
from torchvision.utils import save_image
import torchvision.transforms as T
import cv2
from torch.nn.modules import PairwiseDistance

from scipy.spatial.distance import directed_hausdorff
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fftshift, ifftshift
from pyfftw.interfaces.scipy_fftpack import fft2, ifft2
import cv2
#import monogenic.tools.monogenic_functions as mf
from pyfftw.interfaces.scipy_fftpack import fft2, ifft2
from sklearn.preprocessing import minmax_scale
from AImed.training.utils import *


"""Segmentation model"""
class Segmentation(pl.LightningModule):
    def __init__(
        self,
        model="UTNetV2",
        loss_function="CE+EDGE",
        weight=[0.5, 1, 1, 1],
        lr=0.0005,
        batch_size=8,
    ):
        super().__init__()

        self.epoch = 0

        self.name_list_all = ["_2CH_ED", "_2CH_ES", "_4CH_ED", "_4CH_ES"] * 10
        self.ED = []
        self.ES = []
        self.name = model
        self.weight = weight
        self.flag = None
        self.classes = 3  # [0, 1, 2, 3] for normalizing between 0 and 1
        self.batch_size = batch_size
        self.val_batch_size = 1
        self.loss_function = loss_function
        self.weight = weight
        self.lr = lr

        if model == "UTNetV2":
            self.model = UTNetV2(1, 4)
            self.filepath_logs = "./logs/UTnetv2/"

        if model == "Unet":
            self.model = UNet(1, 4)
            self.filepath_logs = "./logs/Unet/"

        if loss_function == "CE":
            self.criterion = nn.CrossEntropyLoss(weight=torch.tensor(self.weight))
            self.flag = 0

        if loss_function == "CE+EDGE+Dice":
            self.criterion = nn.CrossEntropyLoss(weight=torch.tensor(self.weight))
            self.criterion2 = nn.CrossEntropyLoss(ignore_index=-10)
            self.criterion3 = losses.DiceLoss()
            self.flag = 1

        elif loss_function == "Dice":
            self.criterion = losses.DiceLoss()
            self.flag = 2

        elif loss_function == "Focal":
            self.criterion = losses.FocalLoss(class_num=4, gamma=2)
            self.flag = 3

        elif loss_function == "CE+Dice":
            self.criterion2 = losses.DiceLoss()
            self.criterion = nn.CrossEntropyLoss(weight=torch.tensor(self.weight))
            self.flag = 4

    def multi_class_dice_score(self, img, labels, class_labels=[1,2,3]):
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

    def multi_class_jaccard(self, img, labels, class_labels=[1,2,3]):
          """ Jaccard metric defined for two sets as |A and B| / |A or B|"""

          total_union_volume = 0.
          total_intersect_volume = 0.
          
          outputs = []
          for label in class_labels:
              img_bool = img.flatten() == label
              labels_bool = labels.flatten() == label

              union_volume = sum(img_bool | labels_bool)
              intersect_volume = sum(img_bool & labels_bool)

              total_union_volume += union_volume
              total_intersect_volume += intersect_volume

              outputs.append(intersect_volume/union_volume)

          return total_intersect_volume / total_union_volume, outputs

    def training_step(self, batch, batch_idx):
        img, label = batch

        logits = self.model(img.float())
        # regular weighted CE Loss
        if self.flag == 0:
            train_loss = self.criterion(logits, label.squeeze(1))
            self.log("CE_loss", train_loss)
            self.log("train_loss", train_loss)

        # custom loss incorporating edges
        if self.flag == 1:
            CE_loss = self.criterion(logits, label.squeeze(1))
            # find the edge pixels for the segmentation classes
            x_sobel = K.filters.sobel(label / self.classes)
            reverse = 1.0 - x_sobel
            # set non-edge pixels to -10 such that these will be ignored in the loss
            edges = torch.where(reverse > 0.9989, -10, label).squeeze(1)

            reg_loss = self.criterion2(logits, edges)
            D_loss = self.criterion3(logits, label)
            train_loss = CE_loss + 0.2 * reg_loss + D_loss

            self.log("train_loss", train_loss)
            self.log("CE_loss", CE_loss)
            self.log("reg_loss", reg_loss)
            self.log("train_d_loss", D_loss)

        # Dice loss
        if self.flag == 2:
            train_loss = self.criterion(logits, label)
            self.log("dice_loss", train_loss)
            self.log("train_loss", train_loss)

        # focal loss
        if self.flag == 3:
            train_loss = self.criterion(logits, label.squeeze(1))
            self.log("focal_loss", train_loss)
            self.log("train_loss", train_loss)

        # dice loss + CE loss
        if self.flag == 4:
            CE_loss = self.criterion(logits, label.squeeze(1))
            D_loss = self.criterion2(logits, label)
            train_loss = CE_loss + D_loss
            self.log("train_loss", train_loss)
            self.log("CE_loss", CE_loss)
            self.log("DL_loss", D_loss)

        new = torch.cat((img, label / self.classes), dim=0)
        vis = torchvision.utils.make_grid(new, nrow=self.batch_size, padding=5)
        out_np = K.utils.tensor_to_image(vis)
        # self.logger.experiment["train/segmentations_pairs"].log(File.as_image(out_np))
        return train_loss

    def validation_step(self, batch, batch_idx):
        val_loss = full_validation(self, batch, batch_idx)
        return val_loss

    # validation step without full HD calculation in mm
    #def validation_step(self, batch, batch_idx):
    #    val_loss = full_validation(self, batch, batch_idx)
    #    return val_loss 

    # return output segmenations for EF calculation
    def make_camus_pred_test(self, batch, batch_idx):
        return camus_output_pred_images(self, batch, batch_idx)

    # return output segmentation for validation set
    def make_val_pred(self, batch, batch_idx):
        return val_output_pred_images(self, batch, batch_idx)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)