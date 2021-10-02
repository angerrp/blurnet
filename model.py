import torch.nn as nn
import pytorch_lightning as pl
import torch.optim
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, random_split
from torchmetrics.functional import accuracy
from torchvision import transforms
import torch.nn.functional as F

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
           transforms.RandomCrop(40, padding=4),
           transforms.RandomHorizontalFlip(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        )
        train_dataset = CustomImageDataset(annotations_file="train_set.csv",
                                           img_dir="/home/paul/Downloads/blur-dataset/tmp", transform=trans)
        return DataLoader(train_dataset, batch_size=16, num_workers=4)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.learning_rate)

    def training_step(self, batch, batch_ids):
        x, y = batch
#        x = x.to("cuda")
        pred = self.forward(x)
 #       y = y.to("cuda")
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


train_transforms = torch.nn.Sequential(
    transforms.Resize(300),
    transforms.RandomCrop(50, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
)
test_transforms = torch.nn.Sequential(
    transforms.Resize(300),
    transforms.CenterCrop(300),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
)
train_dataset = CustomImageDataset(annotations_file="train_set.csv", img_dir="/home/paul/Downloads/blur-dataset/tmp")
test_dataset = CustomImageDataset(annotations_file="test_set.csv", img_dir="/home/paul/Downloads/blur-dataset/test", transform=test_transforms)
blur_train, blur_val = random_split(train_dataset, [int(len(train_dataset) * 0.9), len(train_dataset)-int(len(train_dataset) * 0.9)])
blur_train.transforms = train_transforms
blur_val.transforms = test_transforms
train_loader = DataLoader(blur_train, shuffle=True, batch_size=16, pin_memory=True, num_workers=4)
val_loader = DataLoader(blur_val, shuffle=True, batch_size=16, pin_memory=True, num_workers=4)
test_loader = DataLoader(test_dataset, shuffle=True, batch_size=16, pin_memory=True, num_workers=4)


class Blurnet3(pl.LightningModule):
    def __init__(self, lr=0.001):
        super(Blurnet3, self).__init__()
        self.save_hyperparameters()
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
        return DataLoader(blur_train, shuffle=True, batch_size=16, pin_memory=True, num_workers=4)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(test_dataset, shuffle=True, batch_size=16, pin_memory=True, num_workers=4)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(blur_val, shuffle=True, batch_size=16, pin_memory=True, num_workers=4)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.lr,
            momentum=0.9,
            weight_decay=5e-4,
        )
        steps_per_epoch = 45000 // 16
        scheduler_dict = {
            'scheduler': OneCycleLR(
                optimizer,
                0.1,
                epochs=self.trainer.max_epochs,
                steps_per_epoch=steps_per_epoch,
            ),
            'interval': 'step',
        }
        return {'optimizer': optimizer, 'lr_scheduler': scheduler_dict}