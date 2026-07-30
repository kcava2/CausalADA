#Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
#This program is free software;
#you can redistribute it and/or modify
#it under the terms of the MIT License.
#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the MIT License for more details.

import os
import numpy as np
import torch
import torch.utils.data as Data
from PIL import Image
from torchvision import transforms


class PendulumDataset(Data.Dataset):
    def __init__(self, dataset_dir, split="train"):
        self.img_dir = os.path.join(dataset_dir, split)
        self.transform = transforms.ToTensor()
        self.files = [f for f in os.listdir(self.img_dir) if f.endswith('.png')]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        img_path = os.path.join(self.img_dir, fname)
        img = Image.open(img_path).convert('RGBA')
        img_tensor = self.transform(img)  # (4, H, W)

        # filename format: a_{i}_{j}_{shade}_{mid}.png
        parts = fname[:-4].split('_')  # strip .png, split by _
        i_val    = float(parts[1])
        j_val    = float(parts[2])
        shade    = float(parts[3])
        mid      = float(parts[4])
        label = torch.tensor([i_val, j_val, shade, mid], dtype=torch.float32)

        return img_tensor, label


def get_batch_unin_dataset_withlabel(dataset_dir, batch_size, dataset="train"):
    """
    Returns a DataLoader of (image, label) pairs for the pendulum dataset.

    Args:
        dataset_dir: str: root directory containing 'train/' and 'test/' subdirs
        batch_size:  int: batch size
        dataset:     str: 'train' or 'test'
    """
    ds = PendulumDataset(dataset_dir, split=dataset)
    loader = Data.DataLoader(ds, batch_size=batch_size, shuffle=(dataset == "train"), drop_last=True)
    return loader


def _h_A(A, m):
    """
    DAG acyclicity constraint h(A) = tr(e^{A * A}) - m  (NOTEARS).

    Args:
        A: tensor: (m, m): adjacency matrix
        m: int: matrix dimension

    Returns:
        h: scalar tensor: acyclicity penalty (0 iff A is a DAG)
    """
    expm_A = torch.matrix_exp(A * A)
    h = torch.trace(expm_A) - m
    return h
