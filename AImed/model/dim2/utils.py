import torch
import torch.nn as nn
import torch.nn.functional as F
from .conv_layers import (
    BasicBlock,
    Bottleneck,
    SingleConv,
    MBConv,
    FusedMBConv,
    ConvNeXtBlock,
)


def get_block(name):
    block_map = {
        "SingleConv": SingleConv,
        "BasicBlock": BasicBlock,
        "Bottleneck": Bottleneck,
        "MBConv": MBConv,
        "FusedMBConv": FusedMBConv,
        "ConvNeXtBlock": ConvNeXtBlock,
    }
    return block_map[name]


def get_norm(name):
    norm_map = {"bn": nn.BatchNorm3d, "in": nn.InstanceNorm3d}

    return norm_map[name]
