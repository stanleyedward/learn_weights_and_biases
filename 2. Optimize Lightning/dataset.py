from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import torch
import lightning as L


class MNISTDataModule(L.LightningDataModule):
    def __init__(self, batch_size: int, data_dir: str, num_workers: int):
        super().__init__()
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.num_workers = num_workers

        self.num_classes = 10

        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

    def prepare_data(self):
        MNIST(root=self.data_dir, train=True, download=True)
        MNIST(root=self.data_dir, train=False, download=True)

    def setup(self, stage: str):
        if stage == "fit" or None:
            entire_dataset = MNIST(
                root=self.data_dir, train=True, transform=self.transform, download=False
            )
            self.train_dataset, self.val_dataset = random_split(
                dataset=entire_dataset,
                lengths=[55000, 5000],
                generator=torch.Generator().manual_seed(42),
            )

        if stage == "test" or None:
            self.test_dataset = MNIST(
                root=self.data_dir,
                train=False,
                transform=self.transform,
                download=False,
            )

        if stage == "predict" or None:
            self.test_dataset = MNIST(
                root=self.data_dir,
                train=False,
                transform=self.transform,
                download=False,
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
