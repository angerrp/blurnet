import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from dataloader import CustomImageDataset
import matplotlib.pyplot as plt

from model import BlurNet

transforms = torch.nn.Sequential(
    transforms.Resize(300),
    transforms.CenterCrop(300),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
)
#
# transforms = torch.nn.Sequential([
#     # transforms.RandomResizedCrop(224),
#     # transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                          std=[0.229, 0.224, 0.225]),
# ])


def train_loop(train_dataset, optimizer, loss_fn):
    size = len(train_dataset.dataset)
    loader = DataLoader(train_dataset, shuffle=True, batch_size=64, pin_memory=True)
    for batch, (X, y) in enumerate(loader):
       # X = X.unsqueeze(0)
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


dataset = CustomImageDataset(annotations_file="data_binary_classification.csv", img_dir="/home/paul/Downloads/blur-dataset/tmp", transform=transforms)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
model = BlurNet()
model.train()


optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)

loss_fn = nn.CrossEntropyLoss()
epoch = 10
for i in range(epoch):
    train_loop(train_dataset, optimizer, loss_fn)
    torch.save(model.state_dict(), "model.pth")

model = BlurNet()
model.load_state_dict(torch.load("model.pth"))
correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    model.eval()
    loader = DataLoader(test_dataset)

    for data in loader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = model(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))