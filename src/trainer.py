import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from torchvision import transforms
from torchvision import models

from src.dataset import CustomDataset
from src.utils import get_config
from src.train import train, test

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {DEVICE}")
# configs
config = get_config()
batch_size = config["batch_size"]
n_epochs = config["epochs"]
n_classes = config["n_classes"]

DATA_PATH = "../data"
# creating datasets
transform = transforms.Compose(
    [transforms.Resize(size=(224, 224)), transforms.ToTensor()]
)

train_data = CustomDataset("../data/train", transform=transform)
test_data = CustomDataset("../data/test", transform=transform)
val_data = CustomDataset("../data/valid", transform=transform)
trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
valloader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
testloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

model = models.resnet18(weights="DEFAULT")
# for feature extraction, we turn off the grad computation
for param in model.parameters():
    param.requires_grad = False
in_feature = model.fc.in_features
# replace pre-trained fully connected layer with new Linear layer
model.fc = nn.Linear(in_features=in_feature, out_features=n_classes, bias=True)
model.to(DEVICE)
# Gather parameters to be optimized
param_to_update = []
for name, param in model.named_parameters():
    if param.requires_grad:
        param_to_update.append(param)

# optimizers, loss function
optimizer = optim.SGD(param_to_update, lr=0.001, momentum=0.9)
criterion = nn.CrossEntropyLoss()

writer = SummaryWriter()
for epoch in range(n_epochs):
    train_loss, train_acc = train(model, trainloader, criterion, optimizer, DEVICE)
    val_loss, val_acc = test(model, valloader, criterion, DEVICE)
    print(
        f"[{epoch}/{n_epochs-1}]: train_loss: {train_loss}\ttrain_acc: {train_acc}\tval_loss: {val_loss}\tval_acc: {val_acc}"
    )
    # tesnsorboard
    writer.add_scalar("Loss/train", train_loss, epoch)
    writer.add_scalar("Acc/train", train_acc, epoch)
    writer.add_scalar("Loss/val", val_loss, epoch)
    writer.add_scalar("Acc/val", val_acc, epoch)
writer.close()
