# By Matej Jech, Doshisha student ID: evgh3103
# File created for KDD class and for my job as a computer vision researcher/engineer

# Sources:
# Base training loop and data loading: 
# https://towardsdatascience.com/building-a-convolutional-vae-in-pytorch-a0f54c947f71
# Missing pytorch definition due to old version: 
# https://stackoverflow.com/questions/74327447/how-to-use-random-split-with-percentage-split-sum-of-input-lengths-does-not-equ
# ResNet pytorch blocks:
# https://stackoverflow.com/questions/60817390/implementing-a-simple-resnet-block-with-pytorch

import torch
from torch import nn
from torchvision import transforms
from torchmetrics.classification import BinaryAccuracy
import torch.nn.functional as F
from pathlib import Path
import cv2
from skimage import io, transform
import os
import pickle
from torchsummary import summary
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import tqdm


# Definition needed due to older version of Pytorch
import math
from torch import default_generator, randperm
from torch._utils import _accumulate
from torch.utils.data.dataset import Subset
import warnings
def random_split(dataset, lengths, generator=default_generator): 
    r"""
    Randomly split a dataset into non-overlapping new datasets of given lengths.

    If a list of fractions that sum up to 1 is given,
    the lengths will be computed automatically as
    floor(frac * len(dataset)) for each fraction provided.

    After computing the lengths, if there are any remainders, 1 count will be
    distributed in round-robin fashion to the lengths
    until there are no remainders left.

    Optionally fix the generator for reproducible results, e.g.:

    >>> random_split(range(10), [3, 7], generator=torch.Generator().manual_seed(42))
    >>> random_split(range(30), [0.3, 0.3, 0.4], generator=torch.Generator(
    ...   ).manual_seed(42))

    Args:
        dataset (Dataset): Dataset to be split
        lengths (sequence): lengths or fractions of splits to be produced
        generator (Generator): Generator used for the random permutation.
    """
    if math.isclose(sum(lengths), 1) and sum(lengths) <= 1:
        subset_lengths: list[int] = []
        for i, frac in enumerate(lengths):
            if frac < 0 or frac > 1:
                raise ValueError(f"Fraction at index {i} is not between 0 and 1")
            n_items_in_split = int(
                math.floor(len(dataset) * frac)  # type: ignore[arg-type]
            )
            subset_lengths.append(n_items_in_split)
        remainder = len(dataset) - sum(subset_lengths)  # type: ignore[arg-type]
        # add 1 to all the lengths in round-robin fashion until the remainder is 0
        for i in range(remainder):
            idx_to_add_at = i % len(subset_lengths)
            subset_lengths[idx_to_add_at] += 1
        lengths = subset_lengths
        for i, length in enumerate(lengths):
            if length == 0:
                warnings.warn(f"Length of split at index {i} is 0. "
                              f"This might result in an empty dataset.")

    # Cannot verify that dataset is Sized
    if sum(lengths) != len(dataset):    # type: ignore[arg-type]
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    indices = randperm(sum(lengths), generator=generator).tolist()  # type: ignore[call-overload]
    return [Subset(dataset, indices[offset - length : offset]) for offset, length in zip(_accumulate(lengths), lengths)]


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, bias=False)
        else:
            self.skip = None
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels))

    def forward(self, x):
        out = self.block(x)
        out += (x if self.skip is None else self.skip(x))
        if self.in_channels == self.out_channels:
            out = F.sigmoid(out)
        else:
            out = F.relu(out)
        return out


class SegRec7(nn.Module):
    def __init__(self, num_blocks=5, channels_list=[3, 5, 10, 20, 30, 50]):
        super(SegRec7, self).__init__()
        self.blocks = nn.ModuleList()
        self.num_blocks = num_blocks
        self.channels_list = channels_list
        for i in range(self.num_blocks):
            self.blocks.append(ResNetBlock(self.channels_list[i], self.channels_list[i + 1]))
            self.blocks.append(nn.MaxPool2d(kernel_size=2, stride=2, return_indices=False))
        self.flatten = nn.Flatten()
        self.out = nn.Linear(in_features=2200, out_features=1)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.flatten(x)
        x = self.out(x)
        x = F.sigmoid(x)
        return x


class ImageTextDataset(Dataset):
    def __init__(self, dir=Path("data/segrec7"), transform=None):
        self.path_list = os.listdir(dir)
        self.transform = transform
        self.data = []
        for path in tqdm.tqdm(self.path_list):
            truth_label = 1 if 'seg7' in path else 0
            img = cv2.imread(str(dir / path))
            sample = {'image': img, 'label': float(truth_label)}
            if self.transform:
                sample = self.transform(sample)
            self.data.append(sample)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.data[idx]


class Rescale(object):
    """Rescale the image in a sample to a given size.
    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)
        img = transform.resize(image, (new_h, new_w))
        return {'image': img, 'label': label}


class ToTensor(object):
    def __call__(self, sample):
        # numpy image: H x W x C
        # torch image: C x H x W
        image, label = sample['image'].transpose((2, 0, 1)), sample['label']
        return {'image': torch.from_numpy(image).float(), 'label': label}


class SaveBestModel:
    def __init__(self, best_valid_acc=0.0):
        self.best_valid_acc = best_valid_acc
        
    def __call__(self, current_valid_acc, epoch, model, optimizer, metric):
        if current_valid_acc > self.best_valid_acc:
            self.best_valid_acc = current_valid_acc
            print(f"\nBest validation accuracy: {self.best_valid_acc:.2f}")
            print(f"\nSaving best model for epoch: {epoch+1}\n")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metric': metric,
                }, 'model_training/best_segrec7.pth')


def create_data_loaders(dataset_train, dataset_valid, dataset_test, batch_size):
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(dataset_valid, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, valid_loader, test_loader


def save_model(epochs, model, optimizer, criterion):
    print(f"Saving final model...")
    torch.save({
                'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, 'outputs/final_segrec7.pth')


def save_plots(train_acc, valid_acc, train_loss, valid_loss):
    # accuracy plots
    plt.figure(figsize=(10, 7))
    plt.plot(train_acc, color='green', linestyle='-', label='train accuracy')
    plt.plot(valid_acc, color='blue', linestyle='-', label='validataion accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('outputs/accuracy.png')
    
    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(train_loss, color='orange', linestyle='-', label='train loss')
    plt.plot(valid_loss, color='red', linestyle='-', label='validataion loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('outputs/loss.png')


def criterion(preds, golds):
    bce = nn.BCELoss()
    loss = bce(preds.squeeze(), golds)
    return loss


def metric(preds, golds, device):
    binacc = BinaryAccuracy(threshold=0.5).to(device)
    acc = binacc(preds.squeeze(), golds)
    return acc


def train(model, trainloader, optimizer, criterion, metric, device):
    model.train()
    print('Training')
    train_running_loss = 0.0
    train_running_acc = 0.0
    for _, data in tqdm.tqdm(enumerate(trainloader), total=len(trainloader)):
        imgs = data['image'].to(device)
        labels = data['label'].to(device).float()
        out = model(imgs).to(device).float()
        loss = criterion(out, labels)
        acc = metric(out, labels, device)
        train_running_loss += out.shape[0] * loss.item()
        train_running_acc += out.shape[0] * acc.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    epoch_loss = train_running_loss / 1400 # Dataset size is 2000, training is 70% of it
    epoch_acc = train_running_acc / 1400
    return epoch_loss, epoch_acc


def validate(model, validloader, criterion, metric, device):
    model.eval()
    print('Validation')
    valid_running_loss = 0.0
    valid_running_acc = 0.0
    with torch.no_grad():
        for _, data in tqdm.tqdm(enumerate(validloader), total=len(validloader)):
            imgs = data['image'].to(device)
            labels = data['label'].to(device).float()
            out = model(imgs).to(device).float()
            loss = criterion(out, labels)
            acc = metric(out, labels, device)
            valid_running_loss += out.shape[0] * loss.item()
            valid_running_acc += out.shape[0] * acc.item()
        epoch_loss = valid_running_loss / 400 # Dataset size is 2000, validation is 20% of it
        epoch_acc = valid_running_acc / 400
        return epoch_loss, epoch_acc


if __name__ == '__main__':
    scale = Rescale((128, 382))
    totorch = ToTensor()
    textims = ImageTextDataset(dir=Path("data/segrec7"), transform=transforms.Compose([scale,totorch]))
    # textims = torch.load('data/segrec7_dataset.npy')
    # print("Saving dataset")
    # torch.save(textims, 'data/segrec7_dataset.npy', pickle)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)
    train_loss, valid_loss = [], []
    train_acc, valid_acc = [], []

    batch_size = 64
    learning_rate = 1e-3
    epochs = 5

    train_data, valid_data, test_data = random_split(textims, [0.7, 0.2, 0.1], generator=torch.Generator().manual_seed(42))
    train_loader, valid_loader, test_loader = create_data_loaders(train_data, valid_data, test_data, batch_size)

    model = SegRec7().to(device)
    summary(model, input_size=(3, 128, 382), batch_size=batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    save_best_model = SaveBestModel()

    for epoch in range(epochs):
        print(f"[INFO]: Epoch {epoch+1} of {epochs}")
        train_epoch_loss, train_epoch_acc = train(model, train_loader, optimizer, criterion, metric, device)
        valid_epoch_loss, valid_epoch_acc = validate(model, valid_loader, criterion, metric, device)
        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)
        train_acc.append(train_epoch_acc)
        valid_acc.append(valid_epoch_acc)
        print(f"Training loss, accuracy: {train_epoch_loss:.3f}, {train_epoch_acc:.2f}")
        print(f"Validation loss: {valid_epoch_loss:.3f}, {valid_epoch_acc:.2f}")
        save_best_model(valid_epoch_acc, epoch, model, optimizer, metric)
        print('-'*50)

    save_model(epochs, model, optimizer, criterion)
    save_plots(train_acc, valid_acc, train_loss, valid_loss)
    print('TRAINING COMPLETE')