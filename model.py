import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from pytorch_lightning.utilities.types import (EVAL_DATALOADERS,
                                               TRAIN_DATALOADERS)
from torch.utils.data import DataLoader
from torchmetrics.functional import accuracy


class Blurnet(pl.LightningModule):
    """Blur detection model for motion and out of focus blur."""

    def __init__(
        self, train_dataset, validation_dataset, test_dataset, lr=0.01, batch_size=128
    ):
        super().__init__()
        self.learning_rate = lr
        self.batch_size = batch_size

        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.test_dataset = test_dataset

        self.conv1 = nn.Conv2d(3, 96, (7, 7))
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.drop1 = nn.Dropout(0.2)

        self.conv2 = nn.Conv2d(96, 256, (5, 5))
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.drop2 = nn.Dropout(0.2)

        self.full3 = nn.Linear(256 * 20 * 20, 1024)
        self.relu3 = nn.ReLU()
        self.drop3 = nn.Dropout(0.2)

        self.full4 = nn.Linear(1024, 2)
        self.soft4 = nn.Softmax()

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.relu1(x)
        x = self.drop1(x)

        x = self.conv2(x)
        x = self.pool2(x)
        x = self.relu2(x)
        x = self.drop2(x)

        x = x.view(-1, 256 * 20 * 20)

        x = self.full3(x)
        x = self.relu3(x)
        x = self.drop3(x)

        x = self.full4(x)
        x = self.soft4(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log("train_loss", loss)
        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_acc", acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=4,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=4,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.validation_dataset,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=4,
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
