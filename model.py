import numpy as np
import torch.nn as nn
import pytorch_lightning as pl
import torch.optim
from matplotlib import pyplot as plt
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, random_split
from torchmetrics.functional import accuracy
from torchvision import transforms
import torch.nn.functional as F

from dataloader import CustomImageDataset

IM_SIZE = 96

train_transforms = transforms.Compose(
    [
        transforms.Resize(IM_SIZE * 5),
        transforms.RandomCrop(IM_SIZE),

        # transforms.RandomCrop(IM_SIZE, padding=4),
        transforms.ToTensor()
    ]  # transforms.RandomHorizontalFlip(),
    # transforms.RandomCrop(IM_SIZE, padding=4),
    # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
)
test_transforms = transforms.Compose(
    [
        transforms.Resize(IM_SIZE * 5),
        transforms.CenterCrop(IM_SIZE),
        transforms.ToTensor()
    ]
    # transforms.Resize(IM_SIZE*2),
    # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
)

train_dataset = CustomImageDataset(annotations_file="train_set.csv",
                                   img_dir="/home/paul/Downloads/blur-dataset/tmp", transform=train_transforms)
test_dataset = CustomImageDataset(annotations_file="test_set.csv",
                                  img_dir="/home/paul/Downloads/blur-dataset/tmp", transform=test_transforms)
test_set, blur_val = random_split(test_dataset,
                                  [int(len(test_dataset) * 0.9), len(test_dataset) - int(len(test_dataset) * 0.9)])
train_dataset.transforms = train_transforms
blur_val.transforms = test_transforms
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=16, pin_memory=True, num_workers=4)
val_loader = DataLoader(blur_val, batch_size=16, pin_memory=True, num_workers=4)
test_loader = DataLoader(test_set, batch_size=16, pin_memory=True, num_workers=4)

BATCH_SIZE = 128


class Blurnet4(pl.LightningModule):
    def __init__(self, lr=0.01):
        super().__init__()
        self.learning_rate = lr
        # layer 1
        self.conv1 = nn.Conv2d(3, 96, 7)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.drop1 = nn.Dropout(0.2)

        self.conv2 = nn.Conv2d(96, 256, 5)
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
        self.log('train_loss', loss)
        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

        if stage:
            self.log(f'{stage}_loss', loss, prog_bar=True)
            self.log(f'{stage}_acc', acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, 'val')

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, 'test')

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE, pin_memory=True, num_workers=4)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(test_set, batch_size=BATCH_SIZE, pin_memory=True, num_workers=4)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(blur_val, batch_size=BATCH_SIZE, pin_memory=True, num_workers=4)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


class Blurnet3(pl.LightningModule):
    def __init__(self, lr=0.001):
        super(Blurnet3, self).__init__()
        self.learning_rate = lr
        self.save_hyperparameters()
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(32, 64, 5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        # self.conv3 = nn.Conv2d(64, 128, 3)
        # self.pool3 = nn.MaxPool2d(kernel_size=2)
        # self.relu3 = nn.ReLU()

        self.full4 = nn.Linear(64 * 21 * 21, 512)
        self.relu4 = nn.ReLU()
        self.drop4 = nn.Dropout(p=0.5)

        self.full5 = nn.Linear(512, 256)
        self.relu5 = nn.ReLU()
        self.drop5 = nn.Dropout(p=0.5)
        self.full6 = nn.Linear(256, 2)
        self.soft5 = nn.Softmax()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        # x = self.conv3(x)
        # x = self.relu4(x)
        # x = self.pool3(x)
        x = x.view(-1, 64 * 21 * 21)

        x = self.full4(x)
        x = self.relu4(x)
        x = self.drop4(x)

        x = self.full5(x)
        x = self.relu5(x)
        x = self.drop5(x)
        x = self.full6(x)
        x = self.soft5(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log('train_loss', loss)
        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

        if stage:
            self.log(f'{stage}_loss', loss, prog_bar=True)
            self.log(f'{stage}_acc', acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, 'val')

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, 'test')

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE, pin_memory=True, num_workers=4)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(test_set, batch_size=BATCH_SIZE, pin_memory=True, num_workers=4)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(blur_val, batch_size=BATCH_SIZE, pin_memory=True, num_workers=4)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # optimizer = torch.optim.SGD(
        #     self.parameters(),
        #     lr=self.hparams.lr,
        #     momentum=0.9,
        #     weight_decay=5e-4,
        # )
        # optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        # steps_per_epoch = 45000 // 16
        # scheduler_dict = {
        #     'scheduler': OneCycleLR(
        #         optimizer,
        #         0.1,
        #         epochs=self.trainer.max_epochs,
        #         steps_per_epoch=steps_per_epoch,
        #     ),
        #     'interval': 'step',
        # }
        # return {'optimizer': optimizer, 'lr_scheduler': scheduler_dict}
