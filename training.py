import os

import torch
from torch.optim import SGD
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader
from utils.dataset import CustomImageDataset
from torchvision.transforms import ToTensor, Compose, CenterCrop, Resize

from unet_model import UNet

ANNOTATIONS_PATH = os.path.join(".", "data", "annotations.csv")
IMAGES_PATH = os.path.join(".", "data", "train_images")
MASKS_PATH = os.path.join(".", "data", "train_labels")

LR = 1e-2
MOMENTUM = 0.99
INPUT_DIM = 250
OUTPUT_DIM = 52
EPOCHS = 10


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(DEVICE)


def train_loop(_train_dataloader: DataLoader, _model: UNet, _loss_fn: BCEWithLogitsLoss, _optimizer: SGD):
    model.train()
    loss = 0.0

    for data in _train_dataloader:
        images, masks = data["image"].to(DEVICE), data["mask"].to(DEVICE)
        pred = _model(images)
        cost = _loss_fn(pred, masks)

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        loss += cost

    print(f"Train loss: {loss / len(_train_dataloader)}")


if __name__ == "__main__":
    transform = Compose([
        Resize(INPUT_DIM),
        ToTensor()
    ])
    target_transform = Compose([
        Resize(INPUT_DIM),
        CenterCrop(OUTPUT_DIM),
        ToTensor()
    ])

    train_data = CustomImageDataset(ANNOTATIONS_PATH, IMAGES_PATH, MASKS_PATH,
                                    transform=transform, target_transform=target_transform)
    train_dataloader = DataLoader(train_data, batch_size=1, shuffle=True)

    model = UNet()
    model.to(DEVICE)
    loss_fn = BCEWithLogitsLoss()
    optimizer = SGD(model.parameters(), lr=LR, momentum=MOMENTUM)

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)

    print("Done!")
