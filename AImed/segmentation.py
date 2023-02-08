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
from AImed.lvef_estimation import *

"""Segmentation model"""
class Segmentation(pl.LightningModule):
    def __init__(
        self,
        model="UTNetV2",
        loss_function="CE+EDGE",
        weight=[0.5, 1, 1, 1],
        lr=0.0005,
        batch_size=8,
        data = None
    ):
        super().__init__()

        self.epoch = 0
        self.data = data
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

        self.all_views = []
        self.model = model_selection(flag=model).select()
        self.selected_loss = losses.loss_selection(flag = loss_function).select()

    def training_step(self, batch, batch_idx):
        img, label, name, size, EF = batch

        logits = self.model(img.float())

        train_loss = self.selected_loss(logits, label.squeeze(1))
        self.log('train_loss', train_loss)
        for item in self.selected_loss.log_list:
            self.log(item[0], item[1])

        return train_loss


    def validation_step(self, batch, batch_idx):

        img, label, name, size, EF = batch

        logits = self.model(img.float())
        val_loss = self.selected_loss(logits, label.squeeze(1))
        preds = torch.argmax(logits, dim=1)

        
        log_image(self, label, preds)

        for item in self.selected_loss.log_list:
            self.log(item[0], item[1])

        preds, label = get_original_size(preds, label, size)

        if len(self.all_views) < 5:
            self.all_views.append(preds)
         
        # at the beginning the predictions are so bad that EF calculationg
        # goes wrong
        try:  
            _, D_classes = multi_class_dice_score(preds, label.squeeze(1))
            if len(self.all_views) == 4:
                EF, error = calculate_volume(images=self.all_views, true_ef=EF)
                self.all_views = []
                self.log("EF_error", error)
                
        except:
             _, D_classes = multi_class_dice_score(preds, label.squeeze(1))
   
        # log seperate for every class
        for c in range(1,4):
            self.log("D"+str(c-1), D_classes[c-1])
            if D_classes[c-1] != 0:
                self.log("HF"+str(c), compute_HF(label, preds, c))

        self.log("validation/loss", val_loss)
        return val_loss

    # "test step" where the images after the trained model 
    #  are saved for easy access and EF calculation
    def test_step(self, batch, batch_idx):
        img, _, name, size, _ = batch
        logits = self.model(img.float())

        preds = torch.argmax(logits, dim=1)

        preds = get_original_size(preds, None, size)

        if len(self.all_views) < 5:
            self.all_views.append(preds)

        if len(self.all_views) == 4:
            EF = calculate_volume(images=self.all_views, true_ef=None)
            self.all_views = []
            self.log("test/EF", EF)

        # save final segmentations
        torch.save(
            preds, "./test/predictions/"+self.args["model"]+"/" + name
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)