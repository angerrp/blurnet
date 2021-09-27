import torch.nn as nn
import pytorch_lightning as pl
import torch.optim
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
from torch.utils.data import DataLoader
from torchvision import transforms

from dataloader import CustomImageDataset


class BlurNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(32, 64, 5)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(64, 128, 3)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.relu3 = nn.ReLU()

        self.full4 = nn.Linear(128 * 35 * 35, 512)
        self.relu4 = nn.ReLU()
        self.drop4 = nn.Dropout(p=0.5)

        self.full5 = nn.Linear(512, 2)
        self.soft5 = nn.Softmax()

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.pool2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.pool3(x)
        x = self.relu3(x)
        x = x.view(-1, 128 * 35 * 35)
        x = self.full4(x)
        x = self.relu4(x)
        x = self.drop4(x)

        x = self.full5(x)
        x = self.soft5(x)

        return x


class BlurNetLightning(pl.LightningModule):
    def __init__(self, lr=0.001):
        super().__init__()
        self.learning_rate = lr
        self.loss_fn = nn.CrossEntropyLoss()

        self.conv1 = nn.Conv2d(3, 32, 5)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(32, 64, 5)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(64, 128, 3)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.relu3 = nn.ReLU()

        self.full4 = nn.Linear(128 * 35 * 35, 512)
        self.relu4 = nn.ReLU()
        self.drop4 = nn.Dropout(p=0.5)

        self.full5 = nn.Linear(512, 2)
        self.soft5 = nn.Softmax()

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.pool2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.pool3(x)
        x = self.relu3(x)
        x = x.view(-1, 128 * 35 * 35)
        x = self.full4(x)
        x = self.relu4(x)
        x = self.drop4(x)

        x = self.full5(x)
        x = self.soft5(x)

        return x

    def cross_entropy_loss(self, logits, labels):
        return self.loss_fn(logits, labels)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        trans = torch.nn.Sequential(
            transforms.Resize(300),
            transforms.CenterCrop(300),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        )
        train_dataset = CustomImageDataset(annotations_file="data_binary_classification.csv",
                                           img_dir="/home/paul/Downloads/blurr_data/tmp", transform=trans)
        return DataLoader(train_dataset, batch_size=16, num_workers=4)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)

    def training_step(self, batch, batch_ids):
        x, y = batch
        x = x.to("cuda")
        pred = self.forward(x)
        y = y.to("cuda")
        loss = self.loss_fn(pred, y)
        return loss

    def step(self, batch):
        X, y = batch
        yhat = self.forward(X)
        loss = self.loss_fn(yhat, y)
        return y, yhat, loss

    def validation_step(self, batch, batch_idx):
        y, yhat, loss = self.step(batch)
        self.log("val_loss", loss)
