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
from model.dim2.unet import *
from model.dim2.utnetv2 import *
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
from training import augmentation
from training import losses
from pytorch_lightning.loggers import NeptuneLogger
import cv2
import skimage.exposure
import numpy as np
from torchvision.utils import save_image
import torchvision.transforms as T
import cv2

# these remain constant
filepath = "/training/"
filepath_checkpoint = "/model-checkpoints/"

"""Prepare data and create dataset"""
def gaussian_noise(tensor_img, std, mean=0):
    random = torch.randn(tensor_img.shape).to(tensor_img.device) * std + mean
    return tensor_img + random

def brightness_additive(tensor_img, std, mean=0, per_channel=False):
    if per_channel:
        C = tensor_img.shape[1]
    else:
        C = 1

    if len(tensor_img.shape) == 5:
        rand_brightness = torch.normal(mean, std, size=(1, C, 1, 1, 1)).to(
            tensor_img.device
        )
    elif len(tensor_img.shape) == 4:
        rand_brightness = torch.normal(mean, std, size=(1, C, 1, 1)).to(
            tensor_img.device
        )
    else:
        raise ValueError(
            "Invalid input tensor dimension, should be 5d for volume image or 4d for 2d image"
        )
    return tensor_img + rand_brightness


class CAMUSDataset(Dataset):
    def __init__(self, args, mode=None, seed=0, only_quality=True):

        self.mode = mode
        self.doAugmentation = args["augs"]
        self.only_quality = only_quality
        self.args = args
        self.img_slice_list = []
        self.lab_slice_list = []
        self.sizes = []
        self.filepath = "/training/training/"

        # for the official unseen set of camus
        self.filepath_test = "/Copy_of_testing/testing/"

        with open(args["data_root"]+"/list/dataset.yaml", "r") as f:
            img_name_list = yaml.load(f, Loader=yaml.SafeLoader)

        camus_test_names = img_name_list[0:50]

        random.Random(seed).shuffle(img_name_list)

        # img_names consists of all the patient names with either all
        # or only good quality
        img_names = []        
        for patient in tqdm(img_name_list):
            if only_quality:
                with open(
                    args["data_info"]+"/"+patient+"/"+"Info_2CH.cfg"
                ) as info2:
                    i2 = yaml.safe_load(info2)
                    if i2["ImageQuality"] == "Good" or i2["ImageQuality"] == "Medium":
                        img_names.append(patient)
                with open(
                    args["data_info"]+"/"+patient+"/"+"Info_4CH.cfg"
                ) as info4:
                    i4 = yaml.safe_load(info4)
                    if i4["ImageQuality"] == "Good" or i4["ImageQuality"] == "Medium":
                        img_names.append(patient)
            else:
                img_names.append(patient)

        img_names_list = img_names

        test_name_list = img_name_list[: args["test_size"]]
        train_name_list = list(set(img_name_list) - set(test_name_list))

        train_name_list = train_name_list[:2]


        path = './training/training'
        img_list = []
        lab_list = []
        idx = ["_2CH_ED.mhd", "_2CH_ES.mhd", "_4CH_ED.mhd", "_4CH_ES.mhd"]
        if mode == "train":
            # Load training
            print("Load and process training data")
            print(train_name_list)

            for name in tqdm(train_name_list):
                selected_images = idx

                for id in selected_images:
                    img_name = name + id
                    lab_name = name + id.replace(".", "_gt.")

                    itk_img = cv2.resize(
                        sitk.GetArrayFromImage(
                            sitk.ReadImage(
                                os.path.join(path + "/" + name + "/", img_name),
                                sitk.sitkFloat32,
                            )
                        )[0, :, :],
                        dsize=(256, 256),
                        interpolation=cv2.INTER_CUBIC,
                    )
                    itk_lab = cv2.resize(
                        sitk.GetArrayFromImage(
                            sitk.ReadImage(
                                os.path.join(path + "/" + name + "/", lab_name),
                                sitk.sitkFloat32,
                            )
                        )[0, :, :],
                        dsize=(256, 256),
                        interpolation=cv2.INTER_NEAREST,
                    )
                    itk_lab = np.reshape(itk_lab, (1, 256, 256))
                    itk_img = np.reshape(itk_img, (1, 256, 256)) / 255
                    img, lab = self.preprocess(itk_img, itk_lab)
                    img_list.append(img)
                    lab_list.append(lab)

            for i in range(len(img_list)):
                self.img_slice_list.append(img_list[i][0])
                self.lab_slice_list.append(lab_list[i][0])

            if self.doAugmentation:
                print("Augmenting now")
                self.create_augmentations()

            print("Train done, length of dataset:", len(self.img_slice_list))

        ###### actual test set from camus with no labels
        elif mode == "camus":
            self.all_names_save = []
            for name in tqdm(camus_test_names):
                selected_images = idx
                for id in selected_images:
                    img_name = name + id
                    self.all_names_save.append(img_name.replace(".mhd", ""))
                    self.sizes.append(
                        sitk.GetArrayFromImage(
                            sitk.ReadImage(
                                os.path.join(
                                    self.filepath_test + "/" + name + "/", img_name
                                ),
                                sitk.sitkFloat32,
                            )
                        )[0, :, :].shape
                    )
                    itk_img = cv2.resize(
                        sitk.GetArrayFromImage(
                            sitk.ReadImage(
                                os.path.join(
                                    self.filepath_test + "/" + name + "/", img_name
                                ),
                                sitk.sitkFloat32,
                            )
                        )[0, :, :],
                        dsize=(256, 256),
                        interpolation=cv2.INTER_CUBIC,
                    )
                    itk_img = np.reshape(itk_img, (1, 256, 256)) / 255
                    img, lab = self.preprocess(itk_img, itk_img)
                    img_list.append(img)
                    lab_list.append(lab)
            for i in range(len(img_list)):
                self.img_slice_list.append(img_list[i][0])
                self.lab_slice_list.append(lab_list[i][0])

        elif mode == "test":
            print("Load and process test data")
            print(test_name_list)

            self.all_names_save = []
            # Load tests
            for name in tqdm(test_name_list):

                selected_images = idx

                for id in selected_images:

                    img_name = name + id
                    self.all_names_save.append(img_name)
                    lab_name = name + id.replace(".", "_gt.")
                    self.sizes.append(
                        sitk.GetArrayFromImage(
                            sitk.ReadImage(
                                os.path.join(path + "/" + name + "/", lab_name),
                                sitk.sitkFloat32,
                            )
                        )[0, :, :].shape
                    )
                    itk_img = cv2.resize(
                        sitk.GetArrayFromImage(
                            sitk.ReadImage(
                                os.path.join(path + "/" + name + "/", img_name),
                                sitk.sitkFloat32,
                            )
                        )[0, :, :],
                        dsize=(256, 256),
                        interpolation=cv2.INTER_CUBIC,
                    )
                    itk_lab = cv2.resize(
                        sitk.GetArrayFromImage(
                            sitk.ReadImage(
                                os.path.join(path + "/" + name + "/", lab_name),
                                sitk.sitkFloat32,
                            )
                        )[0, :, :],
                        dsize=(256, 256),
                        interpolation=cv2.INTER_NEAREST,
                    )
                    itk_lab = np.reshape(itk_lab, (1, 256, 256))
                    itk_img = np.reshape(itk_img, (1, 256, 256)) / 255

                    img, lab = self.preprocess(itk_img, itk_lab)

                    img_list.append(img)
                    lab_list.append(lab)

            for i in range(len(img_list)):
                self.img_slice_list.append(img_list[i][0])
                self.lab_slice_list.append(lab_list[i][0])

            print("Test done, length of dataset:", len(self.img_slice_list))

    def __len__(self):
        return len(self.img_slice_list)

    def preprocess(self, img, lab):

        import torchvision.transforms as T

        tensor_img = torch.from_numpy(img).float()
        tensor_lab = torch.from_numpy(lab).long()

        return tensor_img, tensor_lab

    def __getitem__(self, idx):
        tensor_img = self.img_slice_list[idx]
        tensor_lab = self.lab_slice_list[idx]
        tensor_img = tensor_img.unsqueeze(0)
        tensor_lab = tensor_lab.unsqueeze(0)
        assert tensor_img.shape == tensor_lab.shape

        return torch.clamp(tensor_img, min=0.0, max=1.0), tensor_lab

    def create_augmentations(self):

        import torchvision.transforms as T

        total = len(self.img_slice_list)
        new_imgs = []
        new_labs = []

        s1 = self.args["s1"]
        s2 = self.args["s2"]
        t = self.args["t"]
        rot = self.args["rotation"]
        for i in range(total):

            ######## affine transformation
            # use the same transformation for ground truth label and image
            affine_transfomer = T.RandomAffine(
                degrees=(-rot, rot), translate=(t, t), scale=(s1, s2)
            )
            new = torch.cat(
                (
                    self.img_slice_list[i].unsqueeze(0),
                    self.lab_slice_list[i].unsqueeze(0),
                ),
                dim=0,
            )
            transformed = affine_transfomer(new.unsqueeze(dim=1))

            img_aug = torch.clamp(transformed[0].squeeze(), min=0.0, max=1.0)
            new_imgs.append(img_aug)
            new_labs.append(transformed[1].squeeze().int())

            ###### SNR augmentation
            if self.args["SNR"] == True:

                # append label to set, this is unchanged
                new_labs.append(self.lab_slice_list[i].int())

                label = self.lab_slice_list[i]
                img = self.img_slice_list[i]

                ##### mask 3 pixels around the triangular border otherwise it will
                ##### be extremely visible in in the monogenic signal
                mask = np.where(label > 0, 1, 0)
                for i in range(256):
                    for j in range(256):
                        if j == 0 and mask[i][j] == 1:
                            mask[i][j + 1] = 0
                        if j < 254:
                            if mask[i][j] == 0:
                                mask[i][j + 1] = 0
                                break
                for i in range(256):
                    for j in range(256):
                        if j < 254:
                            if mask[i][j] == 0 and mask[i][j + 1] == 1:
                                mask[i][j + 1] = 0
                                break
                for i in range(256):
                    for j in range(256):
                        if j < 254:
                            if mask[i][j] == 0 and mask[i][j + 1] == 1:
                                mask[i][j + 1] = 0
                                break
                for i in range(255, -1, -1):
                    for j in range(255, -1, -1):
                        if j == 255 and mask[i][j] == 1:
                            mask[i][j] = 0
                for i in range(255, -1, -1):
                    for j in range(255, -1, -1):
                        if j > 0:
                            if mask[i][j] == 0 and mask[i][j - 1] == 1:
                                mask[i][j - 1] = 0
                                break
                for i in range(255, 252, -1):
                    for j in range(255, -1, -1):
                        mask[i][j] = 0

                rows, cols = img.shape

                # the higher the wavelength and sigma the more "details" are removed
                logGabor, logGabor_H1, logGabor_H2 = mf.monogenic_scale(
                    cols=cols, rows=rows, ss=1, minWaveLength=3, mult=1.8, sigmaOnf=0.2
                )

                IM = fft2(img)
                IMF = IM * logGabor

                IMH1 = IM * logGabor_H1
                IMH2 = IM * logGabor_H2

                f = np.real(ifft2(IMF))
                h1 = np.real(ifft2(IMH1))
                h2 = np.real(ifft2(IMH2))

                ##### LEM
                LEM = torch.FloatTensor(f * f + h1 * h1 + h2 * h2)

                mask = torch.tensor(mask)
                signal = LEM * mask.float()

                #### create gaussian smoothed edges for every ground truth label
                label0 = torch.where(label == 0, -1, 0)
                label0 = torch.where(label0 == -1, 1, 0)  # background
                label1 = torch.where(label == 1, 1, 0)  # big chamber
                label2 = torch.where(label == 2, 1, 0)  # white thing around
                label3 = torch.where(label == 3, 1, 0)  # little one at the bottom

                # blur label 1
                blur = cv2.GaussianBlur(
                    label1.float().numpy(),
                    (0, 0),
                    sigmaX=5,
                    sigmaY=5,
                    borderType=cv2.BORDER_DEFAULT,
                )
                label1 = skimage.exposure.rescale_intensity(
                    blur, in_range=(0.5, 1), out_range=(0, 1)
                )

                # blur label 2
                blur = cv2.GaussianBlur(
                    label2.float().numpy(),
                    (0, 0),
                    sigmaX=5,
                    sigmaY=5,
                    borderType=cv2.BORDER_DEFAULT,
                )
                label2 = skimage.exposure.rescale_intensity(
                    blur, in_range=(0.5, 1), out_range=(0, 1)
                )

                # blur label 3
                blur = cv2.GaussianBlur(
                    label3.float().numpy(),
                    (0, 0),
                    sigmaX=5,
                    sigmaY=5,
                    borderType=cv2.BORDER_DEFAULT,
                )
                label3 = skimage.exposure.rescale_intensity(
                    blur, in_range=(0.5, 1), out_range=(0, 1)
                )

                # blur label 4
                blur = cv2.GaussianBlur(
                    label0.float().numpy(),
                    (0, 0),
                    sigmaX=5,
                    sigmaY=5,
                    borderType=cv2.BORDER_DEFAULT,
                )
                label0 = skimage.exposure.rescale_intensity(
                    blur, in_range=(0.5, 1), out_range=(0, 1)
                )

                # create random scaling factors for every label segmentation seperately
                random_scale = torch.randint(-1, 3, (4,))

                # build up the new signal by multiplying the scaling factors with the EFM
                signal2 = (
                    (signal * random_scale[0] * label0)
                    + (signal * random_scale[1] * label1)
                    + (signal * random_scale[2] * label2)
                    + (signal * random_scale[3] * label3)
                )
                aug = torch.tensor(img) + signal2
                img_aug = torch.clamp(aug, min=0.0, max=1.0)
                new_imgs.append(img_aug)

        for jj in range(len(new_imgs)):
            self.img_slice_list.append(new_imgs[jj])
            self.lab_slice_list.append(new_labs[jj].long())


# datamodule for camus dataset
class CAMUS_DATA(pl.LightningDataModule):
    def __init__(
        self,
        data_root="./tgt_dir",
        t=0.3,
        s1=0.6,
        s2=1.5,
        rotation=20,
        batch_size=8,
        training_size=[256, 256],
        test_size=2,
        SNR=True,
    ):
        super().__init__()

        self.data_root = data_root
        self.batch_size = batch_size
        self.training_size = training_size

        self.args = {
            "data_root": data_root,
            "data_info": "./training/training",
            "training_size": training_size,
            "test_size": test_size,
            "rotation": rotation,
            "s1": s1,
            "s2": s1,
            "t": t,
            "augs": True,
            "SNR": SNR,
        }

    def train_dataloader(self):
        data = CAMUSDataset(self.args, mode="train")
        return DataLoader(data, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        data = CAMUSDataset(self.args, mode="test")
        return DataLoader(data, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        data = CAMUSDataset(self.args, mode="test")
        return DataLoader(data, batch_size=self.batch_size, shuffle=False)

    #### for using the official test set of camus challenge
    # def test_dataloader(self):
    #    data = CAMUSDataset(self.args, mode='camus')
    #    return DataLoader(data, batch_size=self.batch_size, shuffle=False)

"""##**Segmentation model**"""

from torch.nn.modules import PairwiseDistance


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

        if loss_function == "CE+EDGE":
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

        elif loss_function == "DICE+CE":
            self.criterion2 = losses.DiceLoss()
            self.criterion = nn.CrossEntropyLoss(weight=torch.tensor(self.weight))
            self.flag = 4

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

    # runs metrics on validation set
    def test_step(self, batch, batch_idx):
        img = batch[0]
        label = batch[1]
        logits = self.model(img.float())

        if self.flag == 0 or self.flag == 3:
            val_loss = self.criterion(logits, label.squeeze(1))

        if self.flag == 1:
            CE_loss = self.criterion(logits, label.squeeze(1))
            # find the edge pixels for the segmentation classes
            x_sobel = K.filters.sobel(label / self.classes)
            reverse = 1.0 - x_sobel

            # set non-edge pixels to -10 such that these will be ignored in the loss
            edges = torch.where(reverse > 0.9989, -10, label).squeeze(1)
            reg_loss = self.criterion2(logits, edges)

            # visualize the edges
            binary = torch.where(reverse > 0.9989, 0, 1.0)
            out_edge = torchvision.utils.make_grid(
                edges, nrow=self.batch_size, padding=5
            )
            out_np = K.utils.tensor_to_image(out_edge)

            val_loss = CE_loss + 0.2 * reg_loss
            self.log("test_CE_loss", CE_loss)
            self.log("test_reg_loss", reg_loss)

        if self.flag == 2:
            val_loss = self.criterion(logits, label)

        if self.flag == 4:
            CE_loss = self.criterion(logits, label.squeeze(1))
            D_loss = self.criterion2(logits, label)
            val_loss = CE_loss + D_loss
            self.log("val_CE_loss", CE_loss)
            self.log("test_DL_loss", D_loss)

        preds = torch.argmax(logits, dim=1)
        self.log("test_loss", val_loss)

        # need sizes for resize back to original size compute actual distances
        sizes = [
            (843, 512),
            (843, 512),
            (1038, 630),
            (1038, 630),
            (973, 591),
            (973, 591),
            (1038, 630),
            (1038, 630),
            (1038, 630),
            (1038, 630),
            (1038, 630),
            (1038, 630),
            (1103, 669),
            (1103, 669),
            (1103, 669),
            (1103, 669),
            (1232, 748),
            (1232, 748),
            (1232, 748),
            (1232, 748),
            (1038, 630),
            (1038, 630),
            (1038, 630),
            (1038, 630),
            (1038, 630),
            (1038, 630),
            (1038, 630),
            (1038, 630),
            (908, 641),
            (908, 641),
            (908, 641),
            (908, 641),
            (973, 590),
            (973, 590),
            (973, 590),
            (973, 590),
            (778, 472),
            (778, 472),
            (778, 472),
            (778, 472),
        ]

        preds = (
            torch.tensor(
                cv2.resize(
                    preds.squeeze().cpu().numpy(),
                    dsize=(sizes[self.epoch][1], sizes[self.epoch][0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            )
            .float()
            .unsqueeze(0)
        )
        label = (
            torch.tensor(
                cv2.resize(
                    label.squeeze().cpu().numpy(),
                    dsize=(sizes[self.epoch][1], sizes[self.epoch][0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            )
            .int()
            .unsqueeze(0)
        )

        u = torch.argwhere(label.squeeze() == 1).cpu().numpy()
        v = torch.argwhere(preds.squeeze() == 1).cpu().numpy()

        from scipy.spatial.distance import directed_hausdorff

        HF1 = max(directed_hausdorff(u, v)[0], directed_hausdorff(v, u)[0])

        ##### camus dataset-> 0.145x0.154mm^2
        HF1 = HF1 * 0.154

        u = torch.argwhere(label.squeeze() == 2).cpu().numpy()
        v = torch.argwhere(preds.squeeze() == 2).cpu().numpy()

        HF2 = max(directed_hausdorff(u, v)[0], directed_hausdorff(v, u)[0])
        HF2 = HF2 * 0.154

        u = torch.argwhere(label.squeeze() == 3).cpu().numpy()
        v = torch.argwhere(preds.squeeze() == 3).cpu().numpy()

        HF3 = max(directed_hausdorff(u, v)[0], directed_hausdorff(v, u)[0])
        HF3 = HF3 * 0.154

        D, total1 = self.multi_class_dice_score(preds, label.squeeze(1))

        if self.epoch % 4 == 0 or self.epoch % 4 == 2:
            self.ED.append([total1, HF1, HF2, HF3])
        else:
            self.ES.append([total1, HF1, HF2, HF3])

        self.epoch = self.epoch + 1
        return val_loss

    def validation_step(self, batch, batch_idx):
        img = batch[0]
        label = batch[1]
        logits = self.model(img.float())

        if self.flag == 0 or self.flag == 3:
            val_loss = self.criterion(logits.float(), label.squeeze(1))
            self.log("D_loss_val", val_loss)

        if self.flag == 1:
            val_CE_loss = self.criterion(logits, label.squeeze(1))
            # find the edge pixels for the segmentation classes
            x_sobel = K.filters.sobel(label / self.classes)
            reverse = 1.0 - x_sobel

            # set non-edge pixels to -10 such that these will be ignored in the loss
            edges = torch.where(reverse > 0.9989, -10, label).squeeze(1)
            reg_loss = self.criterion2(logits, edges)

            # visualize the edges
            binary = torch.where(reverse > 0.9989, 0, 1.0)
            preds = torch.argmax(logits, dim=1)

            new = torch.cat(
                (preds.unsqueeze(1) / self.classes, reverse / self.classes), dim=0
            )
            vis = torchvision.utils.make_grid(new, nrow=self.batch_size, padding=5)
            out_np = K.utils.tensor_to_image(vis)

            D_loss_val = self.criterion3(logits, label)
            val_loss = val_CE_loss + 0.2 * reg_loss + D_loss_val

            self.log("val_CE_loss", val_CE_loss)
            self.log("val_reg_loss", reg_loss)
            self.log("D_loss_val", D_loss_val)

        if self.flag == 2:
            val_loss = self.criterion(logits, label)
            self.log("D_loss_val", val_loss)

        if self.flag == 4:
            CE_loss = self.criterion(logits, label.squeeze(1))
            D_loss = self.criterion2(logits, label)
            val_loss = CE_loss + D_loss
            self.log("val_CE_loss", CE_loss)
            self.log("D_loss_val", D_loss)

        preds = torch.argmax(logits, dim=1)
        self.log("val_loss", val_loss)



        # log predictions and label
        # new = torch.cat((preds.unsqueeze(1)/self.classes, label/self.classes), dim=0)
        # vis = torchvision.utils.make_grid(new, nrow=self.batch_size, padding=5)
        # out_np = K.utils.tensor_to_image(vis)
        # self.logger.experiment["validation/segmentations_pairs"].log(File.as_image(out_np))

        # log absolute difference
        # vis = torchvision.utils.make_grid((abs(label.squeeze(1)[0].detach()-preds[0].detach()))/self.classes, nrow=self.batch_size, padding=5)
        # out_np = K.utils.tensor_to_image(vis)
        # self.logger.experiment["validation/absolute_differences"].log(File.as_image(out_np))

        return val_loss

    def make_camus_pred_test(self, batch, batch_idx):
        img = batch[0]
        logits = self.model(img.float())

        preds = torch.argmax(logits, dim=1)
        sizes = [
            (1102, 669),
            (1102, 669),
            (1102, 669),
            (1102, 669),
            (778, 499),
            (778, 499),
            (843, 541),
            (843, 541),
            (973, 591),
            (973, 591),
            (1038, 630),
            (1038, 630),
            (1168, 708),
            (1168, 708),
            (1168, 708),
            (1168, 708),
            (908, 551),
            (908, 551),
            (908, 551),
            (908, 551),
            (843, 512),
            (843, 512),
            (843, 512),
            (843, 512),
            (843, 512),
            (843, 512),
            (843, 512),
            (843, 512),
            (908, 551),
            (908, 551),
            (908, 551),
            (908, 551),
            (908, 551),
            (908, 551),
            (908, 551),
            (908, 551),
            (973, 591),
            (973, 591),
            (973, 591),
            (973, 591),
            (1038, 630),
            (1038, 630),
            (1038, 630),
            (1038, 630),
            (908, 551),
            (908, 551),
            (908, 551),
            (908, 551),
            (843, 451),
            (843, 451),
            (843, 451),
            (843, 451),
            (1103, 669),
            (1103, 669),
            (1103, 669),
            (1103, 669),
            (714, 433),
            (714, 433),
            (714, 433),
            (714, 433),
            (1103, 669),
            (1103, 669),
            (1103, 669),
            (1103, 669),
            (1038, 630),
            (1038, 630),
            (1038, 630),
            (1038, 630),
            (1103, 669),
            (1103, 669),
            (1103, 669),
            (1103, 669),
            (1168, 708),
            (1168, 708),
            (1168, 708),
            (1168, 708),
            (1038, 630),
            (1038, 630),
            (973, 590),
            (973, 590),
            (843, 451),
            (843, 451),
            (843, 451),
            (843, 451),
            (1038, 630),
            (1038, 630),
            (1038, 630),
            (1038, 630),
            (908, 582),
            (908, 582),
            (908, 551),
            (908, 551),
            (1103, 669),
            (1103, 669),
            (1103, 669),
            (1103, 669),
            (973, 590),
            (973, 590),
            (973, 590),
            (973, 590),
            (1103, 669),
            (1103, 669),
            (1103, 669),
            (1103, 669),
            (1232, 747),
            (1232, 747),
            (1232, 747),
            (1232, 747),
            (1232, 747),
            (1232, 747),
            (1232, 747),
            (1232, 747),
            (1103, 669),
            (1103, 669),
            (1103, 669),
            (1103, 669),
            (1103, 669),
            (1103, 669),
            (1103, 669),
            (1103, 669),
            (1103, 669),
            (1103, 669),
            (1103, 669),
            (1103, 669),
            (1297, 787),
            (1297, 787),
            (1297, 787),
            (1297, 787),
            (908, 551),
            (908, 551),
            (908, 551),
            (908, 551),
            (1103, 669),
            (1103, 669),
            (1103, 669),
            (1103, 669),
            (1038, 630),
            (1038, 630),
            (1038, 630),
            (1038, 630),
            (1038, 630),
            (1038, 630),
            (973, 590),
            (973, 590),
            (973, 590),
            (973, 590),
            (1038, 630),
            (1038, 630),
            (908, 551),
            (908, 551),
            (908, 551),
            (908, 551),
            (1038, 630),
            (1038, 630),
            (973, 590),
            (973, 590),
            (779, 472),
            (779, 472),
            (779, 472),
            (779, 472),
            (1103, 669),
            (1103, 669),
            (1103, 669),
            (1103, 669),
            (908, 487),
            (908, 487),
            (908, 487),
            (908, 487),
            (1103, 669),
            (1103, 669),
            (1103, 669),
            (1103, 669),
            (973, 590),
            (973, 590),
            (843, 512),
            (843, 512),
            (843, 512),
            (843, 512),
            (843, 512),
            (843, 512),
            (843, 595),
            (843, 595),
            (843, 595),
            (843, 595),
            (1427, 865),
            (1427, 865),
            (1427, 865),
            (1427, 865),
            (584, 354),
            (584, 354),
            (584, 354),
            (584, 354),
            (843, 512),
            (843, 512),
            (843, 512),
            (843, 512),
            (843, 595),
            (843, 595),
            (843, 595),
            (843, 595),
        ]

        preds = (
            torch.tensor(
                cv2.resize(
                    preds.squeeze().cpu().numpy(),
                    dsize=(sizes[self.epoch][1], sizes[self.epoch][0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            )
            .float()
            .unsqueeze(0)
        )
        plt.imshow(preds.squeeze(), cmap="gray")
        plt.show()

        patients = [
            "patient0001_2CH_ED",
            "patient0001_2CH_ES",
            "patient0001_4CH_ED",
            "patient0001_4CH_ES",
            "patient0002_2CH_ED",
            "patient0002_2CH_ES",
            "patient0002_4CH_ED",
            "patient0002_4CH_ES",
            "patient0003_2CH_ED",
            "patient0003_2CH_ES",
            "patient0003_4CH_ED",
            "patient0003_4CH_ES",
            "patient0004_2CH_ED",
            "patient0004_2CH_ES",
            "patient0004_4CH_ED",
            "patient0004_4CH_ES",
            "patient0005_2CH_ED",
            "patient0005_2CH_ES",
            "patient0005_4CH_ED",
            "patient0005_4CH_ES",
            "patient0006_2CH_ED",
            "patient0006_2CH_ES",
            "patient0006_4CH_ED",
            "patient0006_4CH_ES",
            "patient0007_2CH_ED",
            "patient0007_2CH_ES",
            "patient0007_4CH_ED",
            "patient0007_4CH_ES",
            "patient0008_2CH_ED",
            "patient0008_2CH_ES",
            "patient0008_4CH_ED",
            "patient0008_4CH_ES",
            "patient0009_2CH_ED",
            "patient0009_2CH_ES",
            "patient0009_4CH_ED",
            "patient0009_4CH_ES",
            "patient0010_2CH_ED",
            "patient0010_2CH_ES",
            "patient0010_4CH_ED",
            "patient0010_4CH_ES",
            "patient0011_2CH_ED",
            "patient0011_2CH_ES",
            "patient0011_4CH_ED",
            "patient0011_4CH_ES",
            "patient0012_2CH_ED",
            "patient0012_2CH_ES",
            "patient0012_4CH_ED",
            "patient0012_4CH_ES",
            "patient0013_2CH_ED",
            "patient0013_2CH_ES",
            "patient0013_4CH_ED",
            "patient0013_4CH_ES",
            "patient0014_2CH_ED",
            "patient0014_2CH_ES",
            "patient0014_4CH_ED",
            "patient0014_4CH_ES",
            "patient0015_2CH_ED",
            "patient0015_2CH_ES",
            "patient0015_4CH_ED",
            "patient0015_4CH_ES",
            "patient0016_2CH_ED",
            "patient0016_2CH_ES",
            "patient0016_4CH_ED",
            "patient0016_4CH_ES",
            "patient0017_2CH_ED",
            "patient0017_2CH_ES",
            "patient0017_4CH_ED",
            "patient0017_4CH_ES",
            "patient0018_2CH_ED",
            "patient0018_2CH_ES",
            "patient0018_4CH_ED",
            "patient0018_4CH_ES",
            "patient0019_2CH_ED",
            "patient0019_2CH_ES",
            "patient0019_4CH_ED",
            "patient0019_4CH_ES",
            "patient0020_2CH_ED",
            "patient0020_2CH_ES",
            "patient0020_4CH_ED",
            "patient0020_4CH_ES",
            "patient0021_2CH_ED",
            "patient0021_2CH_ES",
            "patient0021_4CH_ED",
            "patient0021_4CH_ES",
            "patient0022_2CH_ED",
            "patient0022_2CH_ES",
            "patient0022_4CH_ED",
            "patient0022_4CH_ES",
            "patient0023_2CH_ED",
            "patient0023_2CH_ES",
            "patient0023_4CH_ED",
            "patient0023_4CH_ES",
            "patient0024_2CH_ED",
            "patient0024_2CH_ES",
            "patient0024_4CH_ED",
            "patient0024_4CH_ES",
            "patient0025_2CH_ED",
            "patient0025_2CH_ES",
            "patient0025_4CH_ED",
            "patient0025_4CH_ES",
            "patient0026_2CH_ED",
            "patient0026_2CH_ES",
            "patient0026_4CH_ED",
            "patient0026_4CH_ES",
            "patient0027_2CH_ED",
            "patient0027_2CH_ES",
            "patient0027_4CH_ED",
            "patient0027_4CH_ES",
            "patient0028_2CH_ED",
            "patient0028_2CH_ES",
            "patient0028_4CH_ED",
            "patient0028_4CH_ES",
            "patient0029_2CH_ED",
            "patient0029_2CH_ES",
            "patient0029_4CH_ED",
            "patient0029_4CH_ES",
            "patient0030_2CH_ED",
            "patient0030_2CH_ES",
            "patient0030_4CH_ED",
            "patient0030_4CH_ES",
            "patient0031_2CH_ED",
            "patient0031_2CH_ES",
            "patient0031_4CH_ED",
            "patient0031_4CH_ES",
            "patient0032_2CH_ED",
            "patient0032_2CH_ES",
            "patient0032_4CH_ED",
            "patient0032_4CH_ES",
            "patient0033_2CH_ED",
            "patient0033_2CH_ES",
            "patient0033_4CH_ED",
            "patient0033_4CH_ES",
            "patient0034_2CH_ED",
            "patient0034_2CH_ES",
            "patient0034_4CH_ED",
            "patient0034_4CH_ES",
            "patient0035_2CH_ED",
            "patient0035_2CH_ES",
            "patient0035_4CH_ED",
            "patient0035_4CH_ES",
            "patient0036_2CH_ED",
            "patient0036_2CH_ES",
            "patient0036_4CH_ED",
            "patient0036_4CH_ES",
            "patient0037_2CH_ED",
            "patient0037_2CH_ES",
            "patient0037_4CH_ED",
            "patient0037_4CH_ES",
            "patient0038_2CH_ED",
            "patient0038_2CH_ES",
            "patient0038_4CH_ED",
            "patient0038_4CH_ES",
            "patient0039_2CH_ED",
            "patient0039_2CH_ES",
            "patient0039_4CH_ED",
            "patient0039_4CH_ES",
            "patient0040_2CH_ED",
            "patient0040_2CH_ES",
            "patient0040_4CH_ED",
            "patient0040_4CH_ES",
            "patient0041_2CH_ED",
            "patient0041_2CH_ES",
            "patient0041_4CH_ED",
            "patient0041_4CH_ES",
            "patient0042_2CH_ED",
            "patient0042_2CH_ES",
            "patient0042_4CH_ED",
            "patient0042_4CH_ES",
            "patient0043_2CH_ED",
            "patient0043_2CH_ES",
            "patient0043_4CH_ED",
            "patient0043_4CH_ES",
            "patient0044_2CH_ED",
            "patient0044_2CH_ES",
            "patient0044_4CH_ED",
            "patient0044_4CH_ES",
            "patient0045_2CH_ED",
            "patient0045_2CH_ES",
            "patient0045_4CH_ED",
            "patient0045_4CH_ES",
            "patient0046_2CH_ED",
            "patient0046_2CH_ES",
            "patient0046_4CH_ED",
            "patient0046_4CH_ES",
            "patient0047_2CH_ED",
            "patient0047_2CH_ES",
            "patient0047_4CH_ED",
            "patient0047_4CH_ES",
            "patient0048_2CH_ED",
            "patient0048_2CH_ES",
            "patient0048_4CH_ED",
            "patient0048_4CH_ES",
            "patient0049_2CH_ED",
            "patient0049_2CH_ES",
            "patient0049_4CH_ED",
            "patient0049_4CH_ES",
            "patient0050_2CH_ED",
            "patient0050_2CH_ES",
            "patient0050_4CH_ED",
            "patient0050_4CH_ES",
        ]
        name = patients[self.epoch]

        torch.save(
            preds, "./predictions/camus_test/" + name
        )

        self.epoch = self.epoch + 1

        return 0

    def make_val_pred(self, batch, batch_idx):
        img = batch[0]
        logits = self.model(img.float())

        from torchvision.utils import save_image

        preds = torch.argmax(logits, dim=1)

        import cv2

        sizes = [
            (843, 512),
            (843, 512),
            (1038, 630),
            (1038, 630),
            (973, 591),
            (973, 591),
            (1038, 630),
            (1038, 630),
            (1038, 630),
            (1038, 630),
            (1038, 630),
            (1038, 630),
            (1103, 669),
            (1103, 669),
            (1103, 669),
            (1103, 669),
            (1232, 748),
            (1232, 748),
            (1232, 748),
            (1232, 748),
            (1038, 630),
            (1038, 630),
            (1038, 630),
            (1038, 630),
            (1038, 630),
            (1038, 630),
            (1038, 630),
            (1038, 630),
            (908, 641),
            (908, 641),
            (908, 641),
            (908, 641),
            (973, 590),
            (973, 590),
            (973, 590),
            (973, 590),
            (778, 472),
            (778, 472),
            (778, 472),
            (778, 472),
        ]

        preds = (
            torch.tensor(
                cv2.resize(
                    preds.squeeze().cpu().numpy(),
                    dsize=(sizes[self.epoch][1], sizes[self.epoch][0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            )
            .float()
            .unsqueeze(0)
        )
        plt.imshow(preds.squeeze(), cmap="gray")
        plt.show()

        patients = [
            "patient0096",
            "patient0096",
            "patient0096",
            "patient0096",
            "patient0105",
            "patient0105",
            "patient0105",
            "patient0105",
            "patient0024",
            "patient0024",
            "patient0024",
            "patient0024",
            "patient0044",
            "patient0044",
            "patient0044",
            "patient0044",
            "patient0031",
            "patient0031",
            "patient0031",
            "patient0031",
            "patient0017",
            "patient0017",
            "patient0017",
            "patient0017",
            "patient0092",
            "patient0092",
            "patient0092",
            "patient0092",
            "patient0004",
            "patient0004",
            "patient0004",
            "patient0004",
            "patient0111",
            "patient0111",
            "patient0111",
            "patient0111",
            "patient0113",
            "patient0113",
            "patient0113",
            "patient0113",
        ]

        idx = ["_2CH_ED.mhd", "_2CH_ES.mhd", "_4CH_ED.mhd", "_4CH_ES.mhd"]
        if self.epoch % 4 == 0:
            name = patients[self.epoch] + "_" + idx[0]

        if self.epoch % 4 == 1:
            name = patients[self.epoch] + "_" + idx[1]

        if self.epoch % 4 == 2:
            name = patients[self.epoch] + "_" + idx[2]

        if self.epoch % 4 == 3:
            name = patients[self.epoch] + "_" + idx[3]

        torch.save(preds, "./predictions/new_unet/" + name)

        self.epoch = self.epoch + 1

        return 0

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)

"""##**Train**"""

# Commented out IPython magic to ensure Python compatibility.
# make use of this implementation for monogenic signals, the paper we cited in
# our paper did not have an implementation unfortunately
#!git clone https://github.com/asp1420/monogenic-cnn-illumination-contrast
# %cd monogenic-cnn-illumination-contrast/

#!pip install pyfftw
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fftshift, ifftshift
from pyfftw.interfaces.scipy_fftpack import fft2, ifft2
import cv2
import monogenic.tools.monogenic_functions as mf
from pyfftw.interfaces.scipy_fftpack import fft2, ifft2
from sklearn.preprocessing import minmax_scale

# train and hyper paramater search
for model_name in ["UTNetV2"]:
    for loss_function in ["CE+EDGE"]:
        for batch_size in [8]:
            for learning_rate in [0.0005, 0.00005]:
                for affine in [
                    [0.7, 1.3, 30, 0.3],
                    [0.7, 1.3, 30, 0.3],
                    [0.9, 1.2, 20, 0.2],
                ]:
                    for SNR in [True, False]:

                        datamodule = CAMUS_DATA(
                            batch_size=batch_size,
                            s1=affine[0],
                            s2=affine[1],
                            rotation=affine[2],
                            t=affine[3],
                            SNR=SNR,
                        )
                        model = Segmentation(
                            model=model_name,
                            loss_function=loss_function,
                            weight=[0.5, 1, 1, 1],
                            lr=learning_rate,
                            batch_size=batch_size,
                        )

                        # for saving best model with lowest validation loss,
                        # a lot of parameters in the name so we dont get confused with if we train
                        # different variants
                        checkpoint_callback = ModelCheckpoint(
                            monitor="D_loss_val",
                            dirpath=filepath_checkpoint,
                            filename=str(model.name)
                            + "-{epoch:02d}-{D_loss_val:.6f}-"
                            + str(model.batch_size)
                            + "-"
                            + str(model.loss_function)
                            + str(affine)
                            + str(learning_rate)
                            + str(SNR)
                            + str(loss_function),
                        )


                        with open('key.txt') as f:
                            key = f.readline()
                        f.close()
                        # plug in your own logger, we used neptune, but we removed our secret api token and project name
                        neptune_logger = NeptuneLogger(
                            api_token=key,
                            project="ace-ch/seg",
                            log_model_checkpoints=False
                        )

                        trainer = Trainer(
                            logger=neptune_logger,
                            accelerator="auto",
                            devices=1 if torch.cuda.is_available() else None,
                            max_epochs=10,
                            callbacks=[checkpoint_callback],
                            log_every_n_steps=1,
                        )

                        # train and validate model
                        trainer.fit(
                            model=model, datamodule=datamodule
                        )  # , ckpt_path="/content/drive/MyDrive/AI4MED/model-checkpoints/UTNetV2-epoch=18-val_loss=0.101841-8-CE+EDGE[0.8, 1.2, 20, 0.2]0.0005TrueCE+EDGE.ckpt")
                        # neptune_logger.experiment.stop()

"""##**Load trained model checks**"""

datamodule = CAMUS_DATA(batch_size=1)
model = Segmentation(
    model="UTNetV2", loss_function="DICE+CE", weight=[0.5, 1, 1, 1], lr=0.0005
)

trainer.test(
    model=model,
    datamodule=datamodule,
    ckpt_path="./model-checkpoints/"
)
###### Other models used in our paper scores ###
# trainer.test(model=model, datamodule = datamodule, ckpt_path="/content/drive/MyDrive/AI4MED/model-checkpoints/Unet-epoch=19-D_loss_val=0.098966-8-DICE+CE[0.9, 1.1, 10, 0.1]0.0005FalseDICE+CE.ckpt")
# BESTE-Unet-epoch=18-D_loss_val=0.095380-8-DICE+CE[0.7, 1.3, 30, 0.3]0.0005TrueDICE+CE
# trainer.test(model=model, datamodule = datamodule, ckpt_path="/content/drive/MyDrive/AI4MED/model-checkpoints/UTNetV2-epoch=18-D_loss_val=0.088782-8-DICE+CE[0.8, 1.2, 20, 0.2]0.0005FalseDICE+CE.ckpt")

##### output ugly metrics for previous models
total_D1 = []
total_D2 = []
total_D3 = []
total_H1 = []
total_H2 = []
total_H3 = []
for item in model.ES:
    D1 = item[0][0]
    D2 = item[0][1]
    D3 = item[0][2]
    H1 = item[1]
    H2 = item[2]
    H3 = item[3]

    total_D1.append(D1)
    total_D2.append(D2)
    total_D3.append(D3)

    total_H1.append(H1)
    total_H2.append(H2)
    total_H3.append(H3)


print(np.mean(total_D1))
print(np.mean(total_D2))
print(np.mean(total_D3))

print(np.mean(total_H1))
print(np.mean(total_H2))
print(np.mean(total_H3))

total_D1 = []
total_D2 = []
total_D3 = []
total_H1 = []
total_H2 = []
total_H3 = []
for item in model.ED:
    D1 = item[0][0]
    D2 = item[0][1]
    D3 = item[0][2]
    H1 = item[1]
    H2 = item[2]
    H3 = item[3]

    total_D1.append(D1)
    total_D2.append(D2)
    total_D3.append(D3)

    total_H1.append(H1)
    total_H2.append(H2)
    total_H3.append(H3)

print(np.mean(total_D1))
print(np.mean(total_D2))
print(np.mean(total_D3))

print(np.mean(total_H1))
print(np.mean(total_H2))
print(np.mean(total_H3))