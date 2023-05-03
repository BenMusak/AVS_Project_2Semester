import os
import utils
import torch
import numpy as np
import torchmetrics
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from utils import plot_confusion_matrix
import torchvision.transforms.functional as TF
from torchvision.io import read_image, ImageReadMode
from torch.utils.data import Dataset, random_split, DataLoader


class GuitarDataset(Dataset):
    def __init__(self, images_dir, csv_dir,transform=None):
        self.images_dir = images_dir
        self.csv_dir = csv_dir 
        self.images_dir_list = os.listdir(images_dir)
        self.transform=transform
        self.df = pd.read_csv(csv_dir)

    def __len__(self):
        return len(self.images_dir_list)

    def __getitem__(self, idx):
        spectogram_path = os.path.join(self.images_dir, self.images_dir_list[idx])
        spectrogram = read_image(spectogram_path, ImageReadMode.GRAY)

        spectrogram = spectrogram.float()
        spectrogram = spectrogram / 255 #Normalize values from [0-255] to [0-1]
        #spectrogram = torch.from_numpy(self.images_dir_list[idx]).float()
        
        for index, row in self.df.iterrows():
            if row['name']==self.images_dir_list[idx][:-3] + 'wav': #The filename has .jpg, and csv has .wav
                label = utils.label_str_to_int(row['guitar_type'])

        return spectrogram, label


class GuitarDataModule(pl.LightningDataModule):
    def __init__(self, images_dir, csv_dir, batch_size=16,transform = None):
        super().__init__()
        self.batch_size = batch_size
        self.all_samples = []

        self.images_dir = images_dir
        self.csv_dir = csv_dir
        self.transform = transform
        
    def prepare_data(self):
        #We don't use this function for loading the data as prepare_data is called from a single GPU. 
        #It can also not be usedto assign state (self.x = y).
        pass
        
    def setup(self, stage=None):
        #Data is loaded from the image and mask directories
        self.all_samples = GuitarDataset(self.images_dir, self.csv_dir,self.transform) 
        #The data is split into train, val and test with a 70/20/10 split
        print("Amount of samples: ", len(self.all_samples))
        train_size = int(len(self.all_samples) * 0.7)
        val_size= int(len(self.all_samples) * 0.2)
        test_size= len(self.all_samples) - train_size - val_size

        print("Train size: ", train_size)
        print("Val size: ", val_size)
        print("Test size: ", test_size)

        self.train_data, self.val_data, self.test_data = random_split(self.all_samples, [train_size, val_size ,test_size ])
        
    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, num_workers=20, pin_memory=True, persistent_workers=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, num_workers=20, pin_memory=True, persistent_workers=True)
    
    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, num_workers=20, pin_memory=True, persistent_workers=True)




class CNNClassifier(pl.LightningModule):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        num_classes = 5
        #In channels, Out channels
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        
        self.fc1 = nn.Linear(128 * 8 * 11, 256) #how to make dynamic :(
        self.fc2 = nn.Linear(256, num_classes)
        
        self.relu = nn.ReLU()
        
        self.criterion = nn.CrossEntropyLoss()
        self.confusion_matrix = torchmetrics.ConfusionMatrix(num_classes=num_classes, task='multiclass')
        
        self.test_confusion_matrix = np.zeros((num_classes, num_classes))       
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        x = self.relu(self.conv3(x))
        x = self.maxpool(x)
        x = self.relu(self.conv4(x))
        x = self.maxpool(x)

        x = x.view(-1, 128 * 8 * 11)        
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss, prog_bar=True, on_epoch=True, on_step = False)
        
        _, predicted = torch.max(y_hat, 1)
        accuracy = (predicted == y).sum().item() / len(y)
        self.log('train_acc', accuracy, prog_bar=True, on_epoch=True, on_step = False)

        
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss, prog_bar=True, on_epoch=True, on_step = False)
        
        _, predicted = torch.max(y_hat, 1)
        accuracy = (predicted == y).sum().item() / len(y)
        self.log('val_acc', accuracy, prog_bar=True, on_epoch=True, on_step = False)

        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        
        loss = self.criterion(y_hat, y)
        self.log('test_loss', loss, prog_bar=True, on_epoch=True, on_step = False)
        
        _, predicted = torch.max(y_hat, 1)
        accuracy = (predicted == y).sum().item() / len(y)
        self.log('test_acc', accuracy, prog_bar=True, on_epoch=True, on_step = False)
        
        confusion_matrix = self.confusion_matrix(predicted, y)
        self.test_confusion_matrix += confusion_matrix.cpu().numpy()
        
    def on_test_epoch_end(self):
        plot_confusion_matrix(self.test_confusion_matrix, ['SC', 'TC', 'SG', 'LP'], filename='confusion_matrix.png')
        
    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x = batch
        y_hat = self(x)
        
        _, predicted = torch.max(y_hat, 1)
        return predicted

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        return optimizer
