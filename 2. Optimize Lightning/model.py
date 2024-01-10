from typing import Any
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
import torch
from torch.nn import Linear, CrossEntropyLoss, functional as F
from torch.optim import Adam
from torchmetrics.functional import accuracy
import lightning as L


class MNIST_LitModule(L.LightningModule):
    def __init__(
        self,
        num_classes: int = 10,
        n_layer_1: int = 128,
        n_layer_2: int = 256,
        learning_rate=1e-3,
    ):
        """method used to define our model parameters"""
        super().__init__()

        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.loss = CrossEntropyLoss()

        self.layer_1 = Linear(in_features=28 * 28, out_features=n_layer_1)
        self.layer_2 = Linear(in_features=n_layer_1, out_features=n_layer_2)
        self.layer_3 = Linear(in_features=n_layer_2, out_features=num_classes)

    def forward(self, x):
        """method used for inference input -> output"""

        batch_size, channels, width, height = x.size()

        # (b, 1, 28, 28) -> (b, 1*28*28)
        x = x.view(batch_size, -1)

        # lets do a  3*(linear+relu)
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = F.relu(self.layer_3(x))

        return x

    def training_step(self, batch, batch_idx):
        """returns a loss from a single batch"""
        _, loss, acc = self._get_preds_loss_accuracy(batch)

        # log loss and matric
        self.log("train_loss", loss)
        self.log("train_accuracy", acc)

        return loss

    def validation_step(self, batch, batch_idx):
        """used for logging metrics"""
        preds, loss, acc = self._get_preds_loss_accuracy(batch)

        # log loss and metric
        self.log("val_loss", loss)
        self.log("val_accuracy", acc)

        return preds

    def test_step(self, batch, batch_idx):
        """used for loggin metrics"""
        _, loss, acc = self._get_preds_loss_accuracy(batch)

        # log loss and metrics
        self.log("test_loss", loss)
        self.log("test_accuracy", acc)

    def configure_optimizers(self):
        """defines model optimizer"""
        return Adam(params=self.parameters(), lr=self.learning_rate)

    def _get_preds_loss_accuracy(self, batch):
        """convenience function since train/valid/test steps are similar"""
        x, y = batch
        logits = self.forward(x)
        preds = torch.argmax(input=logits, dim=1)
        loss = self.loss(logits, y)
        acc = accuracy(preds=preds, target=y, task="multiclass", num_classes=10)

        return preds, loss, acc
