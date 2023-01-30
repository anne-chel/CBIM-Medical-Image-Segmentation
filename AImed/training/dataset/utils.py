import numpy
import SimpleITK as sitk
from pprint import pprint
import matplotlib.pylab as plt


def get_dataset(args, mode, **kwargs):

    if args.dimension == "2d":
        if args.dataset == "acdc":
            from .dim2.dataset_acdc import CMRDataset

            return CMRDataset(
                args,
                mode=mode,
                k_fold=args.k_fold,
                k=kwargs["fold_idx"],
                seed=args.seed,
            )

    else:
        if args.dataset == "acdc":
            from .dim3.dataset_acdc import CMRDataset

            return CMRDataset(
                args,
                mode=mode,
                k_fold=args.k_fold,
                k=kwargs["fold_idx"],
                seed=args.seed,
            )
        elif args.dataset == "lits":
            from .dim3.dataset_lits import LiverDataset

            return LiverDataset(
                args,
                mode=mode,
                k_fold=args.k_fold,
                k=kwargs["fold_idx"],
                seed=args.seed,
            )

        elif args.dataset == "bcv":
            from .dim3.dataset_bcv import BCVDataset

            return BCVDataset(
                args,
                mode=mode,
                k_fold=args.k_fold,
                k=kwargs["fold_idx"],
                seed=args.seed,
            )


def display_image_pair(filename_image, filename_mask, size=(20, 16)):

    if "2CH" in filename_image:
        title = "2 - Chambers"
    else:
        title = "4 - Chambers"

    img = sitk.ReadImage(filename_image)
    img_npa = sitk.GetArrayFromImage(img)
    img_z = int(img.GetDepth() / 2)
    img_npa_zslice = sitk.GetArrayViewFromImage(img)[img_z, :, :]

    label = sitk.ReadImage(filename_mask)
    label_npa = sitk.GetArrayFromImage(label)
    label_z = int(label.GetDepth() / 2)
    label_npa_zslice = sitk.GetArrayViewFromImage(label)[label_z, :, :]

    fig = plt.figure(figsize=size)
    plt.gray()
    plt.axis("off")
    plt.title(title, fontsize=20)

    fig.add_subplot(1, 3, 1)
    plt.imshow(img_npa_zslice)
    plt.title("Raw Image ", fontsize=15)
    plt.axis("off")

    fig.add_subplot(1, 3, 2)
    plt.imshow(label_npa_zslice)
    plt.title("Label", fontsize=15)
    plt.axis("off")

    fig.add_subplot(1, 3, 3)
    plt.imshow(img_npa_zslice)
    plt.imshow(label_npa_zslice, alpha=0.5)
    plt.title("Superposition", fontsize=15)
    plt.axis("off")
