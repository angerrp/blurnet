import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import random_split
from torchvision import transforms

from dataloader import CustomImageDataset
from model import Blurnet

IM_SIZE = 96

train_transforms = transforms.Compose(
    [
        transforms.Resize(IM_SIZE * 5),
        # apply some augmentation
        transforms.RandomCrop(IM_SIZE),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
)
test_transforms = transforms.Compose(
    [
        transforms.Resize(IM_SIZE * 5),
        transforms.CenterCrop(IM_SIZE),
        transforms.ToTensor(),
    ]
)

train_dataset = CustomImageDataset(
    annotations_file="train_set.csv", transform=train_transforms
)
test_dataset = CustomImageDataset(
    annotations_file="test_set.csv", transform=test_transforms
)

# split test set in test and validation (dev) part (90/10)
test_set, validation_dataset = random_split(
    test_dataset,
    [int(len(test_dataset) * 0.9), len(test_dataset) - int(len(test_dataset) * 0.9)],
)
validation_dataset.transforms = test_transforms

# initialize model
model = Blurnet(train_dataset, validation_dataset, test_dataset)

# save best checkpoint
checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    dirpath="./checkpoints",
    filename="sample-blurnet-{epoch:02d}-{val_loss:.2f}",
    save_top_k=1,
    mode="min",
)

# optional logging with wandb, add logger to trainer
# wandb.init(project='blurnet', entity='xxxxx')
# wandb_logger = WandbLogger()

# stop training when validation loss does not decrease for 10 epochs
early_callback = EarlyStopping(
    monitor="val_loss", check_on_train_epoch_end=True, patience=10
)

trainer = pl.Trainer(
    callbacks=[checkpoint_callback, early_callback],
    auto_lr_find=True,
    min_epochs=30,
    log_every_n_steps=2,
)

# search for initial learning rate
trainer.tune(model)
trainer.fit(model)
trainer.test(model)
