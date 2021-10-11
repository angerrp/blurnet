import glob
import os

import PIL.Image
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.pyplot import imshow
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import read_image
from torchvision.utils import make_grid


class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        # image = read_image(img_path)
        image = PIL.Image.open(img_path)

        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


train_transforms = transforms.Compose(
    [
        transforms.Resize(300),
        # transforms.RandomCrop(96),
        transforms.ToTensor(),
        # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),

    ]
    # transforms.Resize(IM_SIZE*2),
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomCrop(IM_SIZE, padding=4),
)

train_dataset = CustomImageDataset(annotations_file="motion_train_set.csv", img_dir="/home/paul/Downloads/blur-dataset/tmp", transform=train_transforms)
image, label = train_dataset[3]
imshow(image.permute(1, 2, 0))
print("")
