import os
import yaml
import glob
import SimpleITK as sitk
from tqdm import tqdm

from utils import ResampleXYZAxis, ResampleLabelToRef


def ResampleCMRImage(
    imImage,
    imLabel,
    im_name,
    lab_name,
    save_path,
    patient_name,
    count,
    target_spacing=(1.0, 1.0, 1.0),
):

    assert imImage.GetSpacing() == imLabel.GetSpacing()
    assert imImage.GetSize() == imLabel.GetSize()

    spacing = imImage.GetSpacing()
    origin = imImage.GetOrigin()

    npimg = sitk.GetArrayFromImage(imImage)
    nplab = sitk.GetArrayFromImage(imLabel)
    z, y, x = npimg.shape

    if not os.path.exists("%s" % (save_path)):
        os.mkdir("%s" % (save_path))

    re_img = ResampleXYZAxis(
        imImage,
        space=(target_spacing[0], target_spacing[1], spacing[2]),
        interp=sitk.sitkBSpline,
    )
    re_lab = ResampleLabelToRef(imLabel, re_img, interp=sitk.sitkNearestNeighbor)

    sitk.WriteImage(re_img, "%s/%s" % (save_path, im_name))
    sitk.WriteImage(re_lab, "%s/%s" % (save_path, lab_name))


if __name__ == "__main__":

    src_path = "/content/drive/MyDrive/AI4MED/training/training"
    tgt_path = "/content/drive/MyDrive/AI4MED/tgt_dir"

    ## Getting patients masks
    mask = []
    for filename in os.listdir(src_path):
        f = os.path.join(src_path, filename)
        for patient in glob.iglob(f"{f}/*_gt.mhd"):
            mask.append(patient)
    mask = sorted(mask)

    ## Getting patients images
    img = []
    for filename in os.listdir(src_path):
        f = os.path.join(src_path, filename)
        for patient in glob.iglob(f"{f}/*.mhd"):
            if ("ED.mhd" in patient) or ("ES.mhd" in patient):
                img.append(patient)
    img = sorted(img)

    idx = [patient.split("/")[7] for patient in img]
    idx = sorted(list(set(idx)))

    if not os.path.exists(tgt_path):
        os.mkdir(tgt_path)

    if not os.path.exists(tgt_path + "/list"):
        os.mkdir("%s/list" % (tgt_path))

    with open("%s/list/dataset.yaml" % tgt_path, "w", encoding="utf-8") as f:
        yaml.dump(idx, f)

    os.chdir(src_path)
    for name in tqdm(os.listdir(".")):
        os.chdir(name)
        count = 0
        for lab_name in os.listdir("."):
            if "_gt.mhd" in lab_name:
                img_name = lab_name.replace("_gt", "")
                img = sitk.ReadImage(img_name)
                lab = sitk.ReadImage(lab_name)

                ResampleCMRImage(
                    img,
                    lab,
                    img_name,
                    lab_name,
                    tgt_path,
                    name,
                    count,
                    (1.5625, 1.5625),
                )
                count += 1
                # print(name, '%d'%count, 'done')

        os.chdir("..")
