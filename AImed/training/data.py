
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
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
import yaml
import random 
from tqdm import tqdm
import SimpleITK as sitk
import os
import cv2
import numpy as np
import torch
from AImed.training.augmentation import create_augmentations
import torchvision.transforms as T
import glob


# Camus dataset module: every image and its label (if present) are preprocessed
# and returns an image and its label, together with its original size and name
class CAMUSDataset(Dataset):
    """ Prequisities: download camus dataset in its original structure and put 
        in correct folder (here: training/camus-dataset). """

    def __init__(self, args, mode=None, seed=0):

        self.args = args
        self.img_slice_list = []
        self.lab_slice_list = []
        self.names = []
        self.sizes = []
        self.true_EF = []
        img_list = []
        lab_list = []
        self.channels = ["_2CH_ED.mhd", "_2CH_ES.mhd", "_4CH_ED.mhd", "_4CH_ES.mhd"]

        ## Getting patient image names from camus dataset
        img = []
        for filename in os.listdir(args["data_root"]):
            if filename not in img:
                img.append(filename)

        img_name_list = sorted(img)

        # shuffle images
        random.Random(seed).shuffle(img_name_list)

        # camus test set without groundtryths are numbers patient1-patient50
        test_imgs = img_name_list[0:51]

        # img_names consists of all the patient names with either all
        # or only good quality
        img_names_list = self.select_quality_samples(img_name_list)

        val_name_list = img_name_list[: args["test_size"]]
        train_name_list = list(set(img_name_list) - set(val_name_list))

        train_name_list = train_name_list[:2]

        if mode == "train":
            self.select_true_EF(train_name_list)
            self.create_list(train_name_list, img_list, lab_list)
            if args['augs']:
                create_augmentations(self)

        elif mode == "val":
            self.select_true_EF(val_name_list)
            self.create_list(val_name_list, img_list, lab_list)

        ###### actual test set from camus with no labels
        elif mode == "test":
            self.create_list_test(test_imgs, img_list)
            # dummy ef for easy handling in dataloader
            for i in len(test_imgs):
                self.true_EF.append([0])

    def select_true_EF(self, img_name_list):
        for patient in img_name_list:
            with open(
                self.args["data_root"]+"/"+patient+"/"+"Info_2CH.cfg"
            ) as info2:
                i2 = yaml.safe_load(info2)

                # append twice for ES and ED
                self.true_EF.append(i2["LVef"])
                self.true_EF.append(i2["LVef"])
            with open(
                self.args["data_root"]+"/"+patient+"/"+"Info_4CH.cfg"
            ) as info4:
                i4 = yaml.safe_load(info4)
                self.true_EF.append(i4["LVef"])
                self.true_EF.append(i4["LVef"])

    def select_quality_samples(self, img_name_list):

        img_names = []        
        for patient in img_name_list:
            if self.args["only_quality"]:
                with open(
                    self.args["data_root"]+"/"+patient+"/"+"Info_2CH.cfg"
                ) as info2:
                    i2 = yaml.safe_load(info2)
                    if i2["ImageQuality"] == "Good" or i2["ImageQuality"] == "Medium":
                        img_names.append(patient)
                with open(
                    self.args["data_root"]+"/"+patient+"/"+"Info_4CH.cfg"
                ) as info4:
                    i4 = yaml.safe_load(info4)
                    if i4["ImageQuality"] == "Good" or i4["ImageQuality"] == "Medium":
                        img_names.append(patient)
            else:
                img_names.append(patient)
        return img_names

    # return list for preprocessed images and their labels
    def create_list(self, name_list, img_list, lab_list):
        # Load tests

        for name in name_list:

            for id in self.channels:

                img_name = name + id
                lab_name = name + id.replace(".", "_gt.")

                self.sizes.append(self.get_size(name, img_name))
                self.names.append(img_name)

                img = self.preprocess_img(name, img_name, self.args["data_root"])
                lab = self.preprocess_lab(name, lab_name)

                img_list.append(img)
                lab_list.append(lab)

        for i in range(len(img_list)):
            self.img_slice_list.append(img_list[i][0])
            self.lab_slice_list.append(lab_list[i][0])

    def create_list_test(self, name_list, img_list):

        for name in name_list:
            for id in self.channels:

                img_name = name + id

                self.sizes.append(self.get_size(name, img_name))
                self.names.append(img_name)

                img = self.preprocess_img(name, img_name, self.args["data_root_test"])

                img_list.append(img)

        for i in range(len(img_list)):
            self.img_slice_list.append(img_list[i][0])

            # dummy label image for simple use in dataloader
            self.lab_slice_list.append([0])

    # preprocess image
    def preprocess_img(self, name, img_name, data_root):

        itk_img = cv2.resize(
            sitk.GetArrayFromImage(
                sitk.ReadImage(
                    os.path.join(data_root + "/" + name + "/", img_name),
                    sitk.sitkFloat32,
                )
            )[0, :, :],
            dsize=(256, 256),
            interpolation=cv2.INTER_CUBIC,
        )

        itk_img = np.reshape(itk_img, (1, 256, 256)) / 255
        tensor_img = torch.from_numpy(itk_img).float()        
        return tensor_img

    # preprocess ground truth label
    def preprocess_lab(self, name, lab_name):
        itk_lab = cv2.resize(
            sitk.GetArrayFromImage(
                sitk.ReadImage(
                    os.path.join(self.args["data_root"] + "/" + name + "/", lab_name),
                    sitk.sitkFloat32,
                )
            )[0, :, :],
            dsize=(256, 256),
            interpolation=cv2.INTER_NEAREST,
        )
        itk_lab = np.reshape(itk_lab, (1, 256, 256))
        tensor_lab = torch.from_numpy(itk_lab).long()
        return tensor_lab

    # get original size of image
    def get_size(self, name, img_name):

        return sitk.GetArrayFromImage(
                    sitk.ReadImage(
                        os.path.join(
                            self.args["data_root"] + "/" + name + "/", img_name
                        ),
                        sitk.sitkFloat32,
                    )
                )[0, :, :].shape

    # get length of dataset
    def __len__(self):
        return len(self.img_slice_list)

    # return img, label, name,original size and true EF
    def __getitem__(self, idx):
        tensor_img = self.img_slice_list[idx]
        tensor_lab = self.lab_slice_list[idx]
        tensor_img = tensor_img.unsqueeze(0)
        tensor_lab = tensor_lab.unsqueeze(0)
        assert tensor_img.shape == tensor_lab.shape
        return torch.clamp(tensor_img, min=0.0, max=1.0), tensor_lab, self.names[idx], self.sizes[idx], self.true_EF[idx]

# datamodule for camus dataset
class CAMUS_DATA(pl.LightningDataModule):
    def __init__(
        self,
        data_root=None,
        data_root_test = None,
        t=0.3,
        s1=0.6,
        s2=1.5,
        rotation=20,
        batch_size=8,
        training_size=[256, 256],
        test_size=1,
        SNR=True,
        aff=True,
        only_quality=True
    ):
        super().__init__()

        self.data_root = data_root
        self.data_root_test = data_root_test
        self.batch_size = batch_size
        self.training_size = training_size

        self.args = {
            "data_root": data_root,
            "data_root_test":data_root_test,
            "training_size": training_size,
            "test_size": test_size,
            "rotation": rotation,
            "s1": s1,
            "s2": s1,
            "t": t,
            "augs": True,
            "SNR": SNR,
            "aff": aff,
            "only_quality": only_quality
        }

    def train_dataloader(self):
        data = CAMUSDataset(self.args, mode="train")
        return DataLoader(data, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        data = CAMUSDataset(self.args, mode="val")
        return DataLoader(data, batch_size=1, shuffle=False)

    def test_dataloader(self):
        data = CAMUSDataset(self.args, mode="test")
        return DataLoader(data, batch_size=1, shuffle=False)