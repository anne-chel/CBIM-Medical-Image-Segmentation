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

def quick_validation(self, batch, batch_idx):
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

def full_validation(self, batch, batch_idx):
    img = batch[0]
    label = batch[1]
    logits = self.model(img.float())

    if self.flag == 0 or self.flag == 3:
        val_loss = self.criterion(logits, label.squeeze(1))
        self.log("CE_loss", val_loss)
        self.log("D_loss_val", val_loss)

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
        self.log("D_loss_val", val_loss)

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

def camus_output_pred_images(self, batch, batch_idx):
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
    # !!!!!! to do save it properly !!!!!!
    #plt.imshow(preds.squeeze(), cmap="gray")
    #plt.show()

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

def val_output_pred_images(self, batch, batch_idx):
    img = batch[0]
    logits = self.model(img.float())
    preds = torch.argmax(logits, dim=1)

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