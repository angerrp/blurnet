import torch
import wandb
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, random_split

from dataloader import CustomImageDataset
from model import Blurnet3, Blurnet4
import pytorch_lightning as pl
from torchvision import transforms

transforms = torch.nn.Sequential(
    # transforms.RandomCrop(50, padding=
    transforms.Resize(50),
    transforms.CenterCrop(50),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
)
train_dataset = CustomImageDataset(annotations_file="train_set.csv", img_dir="/home/paul/Downloads/blur-dataset/tmp")
test_dataset = CustomImageDataset(annotations_file="test_set.csv", img_dir="/home/paul/Downloads/blur-dataset/test")
blur_train, blur_val = random_split(train_dataset,
                                    [int(len(train_dataset) * 0.9), len(train_dataset) - int(len(train_dataset) * 0.9)])
blur_train.transforms = transforms
train_loader = DataLoader(blur_train, shuffle=True, batch_size=16, pin_memory=True, num_workers=4)
val_loader = DataLoader(blur_val, shuffle=True, batch_size=16, pin_memory=True, num_workers=4)
test_loader = DataLoader(test_dataset, shuffle=True, batch_size=16, pin_memory=True, num_workers=4)

checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    dirpath="./checkpoints",
    filename="sample-blurnet-{epoch:02d}-{val_loss:.2f}",
    save_top_k=1,
    mode="min",
)
wandb.init(project='blurnet', entity='etimoz')
wandb_logger = WandbLogger()  # newline 2
checkpoint = ModelCheckpoint(monitor="val_loss", mode="min")
early_callback = EarlyStopping(monitor="val_loss", check_on_train_epoch_end=True, patience=10)
trainer = pl.Trainer(callbacks=[checkpoint_callback, early_callback], auto_lr_find=True, logger=wandb_logger, min_epochs=30,
                     log_every_n_steps=2)
# trainer.tune(model)

# model = Blurnet3()
model = Blurnet4()
trainer.tune(model)
trainer.fit(model)
trainer.test(model)
