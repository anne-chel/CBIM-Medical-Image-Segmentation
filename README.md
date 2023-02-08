## Camus Segmentation Challenge AI for Medical Imaging
Group names: Anne Chel, Victor Kyriacou, Vıctor Retamal Guiberteau, Laura Latorre Moreno

This forked repository contains the work of our group, build on top of the package CBIM-Medical-Image-Segmentation.

Segmentation and EF calculation.

### Environment
To run the env creator in Snellius, clone this repository and ```cd ./CBIM-Medical-Image-Segmentation```. Run ```sbatch env_build.sh``` in the terminal and after approx 30 mins, the environment should be created. 

To run locally: ```conda env -f create environment.yml``` and ```conda activate camus``` 

### Data
Download official CAMUS dataset from https://humanheart-project.creatis.insa-lyon.fr/database/#collection/6373703d73e9f0047faa1bc8 and put with its original structure and quality information in training/camus-dataset. Download official test dataset from camus and place in testing/camus-test-dataset

### Currently Supporting models
- [UTNetV2](https://arxiv.org/abs/2203.00131) (Official implementation)
- [UNet] Including 2D, 3D with different building block, e.g. double conv, Residual BasicBlock, Bottleneck, MBConv, or ConvNeXt block.

### Supporting Datasets
- CAMUS 

### To do
- signifance testing

### Citation
```
@inproceedings{gao2021utnet,
  title={UTNet: a hybrid transformer architecture for medical image segmentation},
  author={Gao, Yunhe and Zhou, Mu and Metaxas, Dimitris N},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={61--71},
  year={2021},
  organization={Springer}
}

@misc{gao2022datascalable,
      title={A Data-scalable Transformer for Medical Image Segmentation: Architecture, Model Efficiency, and Benchmark}, 
      author={Yunhe Gao and Mu Zhou and Di Liu and Zhennan Yan and Shaoting Zhang and Dimitris N. Metaxas},
      year={2022},
      eprint={2203.00131},
      archivePrefix={arXiv},
      primaryClass={eess.IV}
}

```
