from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
def get_data():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # resize to match ResNet input
        transforms.Grayscale(num_output_channels=3),  # convert to 3 channels
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)  # normalize RGB channels
    ])
    training = datasets.ImageFolder('data/training/', transform=transform)
    test = datasets.ImageFolder('data/testing/', transform=transform)

    train_loader = DataLoader(training, batch_size=64, shuffle=True)
    test_loader = DataLoader(test, batch_size=64, shuffle=False)

    return train_loader, test_loader
