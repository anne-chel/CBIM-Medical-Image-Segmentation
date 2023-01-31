
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

"""Prepare data and create dataset"""
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

        # unseen validation set is constant for now
        camus_test_names = img_name_list[0:30]

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
        #train_name_list = train_name_list[:2]


        path = './AImed/training/training'
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
                create_augmentations(self)
                #self.create_augmentations()

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

# datamodule for camus dataset
class CAMUS_DATA(pl.LightningDataModule):
    def __init__(
        self,
        data_root="./AImed/tgt_dir",
        t=0.3,
        s1=0.6,
        s2=1.5,
        rotation=20,
        batch_size=8,
        training_size=[256, 256],
        test_size=20,
        SNR=True,
    ):
        super().__init__()

        self.data_root = data_root
        self.batch_size = batch_size
        self.training_size = training_size

        self.args = {
            "data_root": data_root,
            "data_info": "./AImed/training/training",
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