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


class LitModel(L.LightningModule):
    def __init__(
        self, input_shape, num_classes: int, learning_rate=2e-4, transfer: bool = True
    ):
        super().__init__()

        # log hparams
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.input_shape = input_shape
        self.num_classes = num_classes

        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.loss = nn.CrossEntropyLoss()

        self.resnet18_weights = models.ResNet18_Weights.DEFAULT
        self.resnet18_transforms = self.resnet18_weights.transforms()
        self.resnet18_model = models.resnet18(weights=self.resnet18_weights)

        # freeze params
        if transfer:
            for params in self.resnet18_model.parameters():
                params.requires_grad = False

        n_sizes = self._get_conv_output(shape=input_shape)

        self.classifier = nn.Linear(in_features=n_sizes, out_features=num_classes)
        
        self.test_step_outputs = []

    def _get_conv_output(self, shape):
        batch_size = 1
        tmp_input = torch.autograd.Variable(torch.rand(batch_size, *shape))

        output_feat = self._forward_features(tmp_input)
        n_size = output_feat.data.view(batch_size, -1).size(1)
        return n_size

    def _forward_features(self, x):
        x = self.resnet18_model(x)
        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self.forward(x)
        loss = self.loss(out, y)
        acc = self.accuracy(out, y)

        self.log("train/loss", loss)
        self.log("train/acc", acc)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self.forward(x)
        loss = self.loss(out, y)
        acc = self.accuracy(out, y)

        self.log("val/loss", loss)
        self.log("val/acc", acc)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        out = self.forward(x)
        loss = self.loss(out, y)

        self.test_step_outputs.append({"loss": loss, "outputs": out, "y": y})
    
        return loss

    def on_test_epoch_end(self):
        loss = torch.stack([x["loss"] for x in self.test_step_outputs]).mean()
        output = torch.cat([x["outputs"] for x in self.test_step_outputs], dim=0)

        y = torch.cat([x["y"] for x in self.test_step_outputs], dim=0)

        self.log("test/loss", loss)
        acc = self.accuracy(output, y)
        self.log("test/acc", acc)

        self.test_y = y
        self.test_output = output
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
