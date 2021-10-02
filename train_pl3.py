import torch
import wandb
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, random_split

from dataloader import CustomImageDataset
from model import BlurNetLightning, Blurnet3
import pytorch_lightning as pl
from torchvision import transforms

model = Blurnet3()

transforms = torch.nn.Sequential(
    transforms.Resize(300),
    transforms.RandomCrop(50, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
)
train_dataset = CustomImageDataset(annotations_file="train_set.csv", img_dir="/home/paul/Downloads/blur-dataset/tmp")
test_dataset = CustomImageDataset(annotations_file="test_set.csv", img_dir="/home/paul/Downloads/blur-dataset/test")
blur_train, blur_val = random_split(train_dataset, [int(len(train_dataset) * 0.9), len(train_dataset)-int(len(train_dataset) * 0.9)])
blur_train.transforms = transforms
train_loader = DataLoader(blur_train, shuffle=True, batch_size=16, pin_memory=True, num_workers=4)
val_loader = DataLoader(blur_val, shuffle=True, batch_size=16, pin_memory=True, num_workers=4)
test_loader = DataLoader(test_dataset, shuffle=True, batch_size=16, pin_memory=True, num_workers=4)


checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    dirpath="./checkpoints",
    filename="sample-blurnet-{epoch:02d}-{val_loss:.2f}",
    save_top_k=3,
    mode="min",
)
wandb.init(project='blurnet', entity='etimoz')
wandb_logger = WandbLogger()  # newline 2

trainer = pl.Trainer(progress_bar_refresh_rate=20, callbacks=[EarlyStopping(monitor="val_loss", patience=3), checkpoint_callback], auto_lr_find=True, logger=wandb_logger)
#trainer.tune(model)

trainer.fit(model, train_loader, val_loader)
trainer.test(model, test_loader)