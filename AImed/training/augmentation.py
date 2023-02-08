import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import pdb
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fftshift, ifftshift
from pyfftw.interfaces.scipy_fftpack import fft2, ifft2
import cv2
import AImed.monogenic.tools.monogenic_functions as mf
from pyfftw.interfaces.scipy_fftpack import fft2, ifft2
from sklearn.preprocessing import minmax_scale
import skimage.exposure
import torchvision.transforms as T

# SNR augmentation and affine transformation
# possible extend?
def create_augmentations(self):      

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

            if self.args['aff'] == True:
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
