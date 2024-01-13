import os
from typing import Any

import lightning as L

# your favorite machine learning tracking tool
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities.types import STEP_OUTPUT

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import random_split, DataLoader

from torchmetrics import Accuracy

from torchvision import transforms
from torchvision.datasets import StanfordCars
from torchvision.datasets.utils import download_url
import torchvision.models as models

import wandb


class StanfordCarsDataModule(L.LightningDataModule):
    def __init__(self, batch_size, model_transforms: transforms, data_dir: str = "./"):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

        # Augmentation policy for training set
        self.augmentation = transforms.Compose(
            [
                transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
                transforms.RandomRotation(degrees=15),
                transforms.RandomHorizontalFlip(),
                model_transforms,
                transforms.ToTensor(),
            ]
        )
        # Preprocessing steps applied to validation and test set.
        self.transform = transforms.Compose([model_transforms, transforms.ToTensor()])

        self.num_classes = 196

    def prepare_data(self):
        StanfordCars(root=self.data_dir, download=True, split="train")
        StanfordCars(root=self.data_dir, download=True, split="test")

    def setup(self, stage=None):
        entire_dataset = StanfordCars(root=self.data_dir, download=False, split="train")
        # split dataset
        self.train, self.val = random_split(
            dataset=entire_dataset,
            lengths=[6500, 1644],
            generator=torch.Generator().manual_seed(seed=42),
        )

        self.test = StanfordCars(root=self.data_dir, download=True, split="test")

        self.test = random_split(self.test, [len(self.test)])[0]

        self.train.dataset.transform = self.augmentation
        self.val.dataset.transform = self.transform
        self.test.dataset.transform = self.transform

    def train_dataloader(self):
        return DataLoader(
            self.train, batch_size=self.batch_size, shuffle=True, num_workers=4
        )

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=4)
