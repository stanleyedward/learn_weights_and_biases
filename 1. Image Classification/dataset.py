import lightning.pytorch as L
import torch
from torch.utils.data import random_split, DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10


class CIFAR10DataModule(L.LightningDataModule):
    def __init__(self, batch_size: int, data_dir: str, num_workers: int):
        super().__init__()
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.num_workers = num_workers

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        self.num_classes = 10

    def prepare_data(self):
        CIFAR10(root=self.data_dir, train=True, download=True)
        CIFAR10(root=self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            entire_dataset = CIFAR10(
                root=self.data_dir, train=True, download=False, transform=self.transform
            )
            self.train_dataset, self.val_dataset = random_split(
                dataset=entire_dataset,
                lengths=[45000, 5000],
                generator=torch.Generator().manual_seed(42),
            )
        if stage == "test" or stage is None:
            self.test_dataset = CIFAR10(
                root=self.data_dir, train=False, download=False, transform=self.transform
            )

        if stage == "predict" or stage is None:
            self.test_dataset = CIFAR10(
                root=self.data_dir, train=False, download=False, transform=self.transform
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
