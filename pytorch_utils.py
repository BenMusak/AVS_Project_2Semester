import os
import utils
import wandb
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
        #return len(self.images_dir_list)
        return self.df.shape[0]

    def __getitem__(self, idx):
        #spectogram_path = os.path.join(self.images_dir, self.images_dir_list[idx])
        spectogram_path = os.path.join(self.images_dir, self.df.iloc[idx]['name'])
        list=[]
        #try:
        spectrogram = read_image(spectogram_path, ImageReadMode.GRAY)
        #except:
        #    #print(spectogram_path+" is fucked up")
        #    list.append(spectogram_path)
        #    df = pd.read_csv('metadata.csv')
        #    df = df[df['name'] != self.df.iloc[idx]['name']]
        #    df.to_csv('metadata.csv', index=False)
        #    spectrogram = read_image('/home/ubuntu/Guitar_Data_Mel_Spectograms/AudioStrumming_Gibson_honeyburst_LP_Neck_Amajor_JM_26.png', ImageReadMode.GRAY)
        spectrogram = spectrogram.float()
        spectrogram = spectrogram / 255 #Normalize values from [0-255] to [0-1]
        #spectrogram = torch.from_numpy(self.images_dir_list[idx]).float()
        #print('Spectrogram name: ', self.df.iloc[idx]['name'])
        row = self.df.iloc[idx]
        # for index, row in self.df.iterrows():
        #     if row['name']==self.images_dir_list[idx][:-3] + 'wav': #The filename has .jpg, and csv has .wav
        #print("Row: ", row)
        manufacturer = utils.manufacturer_str_to_int(row['manufacturer'])
        #print("Manu and row:", manufacturer,row['manufacturer'])
        guitar_type = utils.guitar_type_str_to_int(row['guitar_type'])
        pickup =utils.pickup_str_to_int(row['pickup'])
        pickup_position = utils.pickup_position_str_to_int(row['pickup_position'])
        strumming= utils.strumming_str_to_int(row['strumming'])
        player = utils.player_str_to_int(row['player'])
        #print("Manu:", manufacturer)
        return spectrogram, {
                                'manufacturer': manufacturer,
                                'guitar_type': guitar_type,
                                'pickup': pickup,
                                'pickup_position': pickup_position,
                                'strumming' : strumming,
                                'player': player    
                            }


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

        self.train_data, self.val_data, self.test_data = random_split(self.all_samples, [train_size, val_size ,test_size ], generator=torch.Generator().manual_seed(69))
        
    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, num_workers=20, pin_memory=True, persistent_workers=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, num_workers=20, pin_memory=True, persistent_workers=True)
    
    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, num_workers=20, pin_memory=True, persistent_workers=True)




class CNNClassifier(pl.LightningModule):
    def __init__(self,optimizer_str,lr):
        super(CNNClassifier, self).__init__()
        self.lr=lr
        self.optimizer_str=optimizer_str
        #In channels, Out channels
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=0)  
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        #self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        
        self.fc1 = nn.Linear(64 * 61 * 91, 128)
        self.fc2 = nn.Linear(128, 126)

        self.manufacturer = nn.Linear(126, 7)
        self.guitar_type = nn.Linear(126, 4)
        self.pickup = nn.Linear(126, 2)
        self.pickup_position = nn.Linear(126, 3)
        self.strumming = nn.Linear(126, 2)
        self.player = nn.Linear(126, 6)
        
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

        self.criterion = nn.CrossEntropyLoss()
        
        self.confmat_manufacturer = torchmetrics.ConfusionMatrix(num_classes = 7, task='multiclass') 
        self.confmat_guitar_type = torchmetrics.ConfusionMatrix(num_classes = 4, task='multiclass')
        self.confmat_pickup = torchmetrics.ConfusionMatrix(num_classes = 2, task='multiclass')
        self.confmat_pickup_position = torchmetrics.ConfusionMatrix(num_classes = 3, task='multiclass')
        self.confmat_strumming = torchmetrics.ConfusionMatrix(num_classes = 2, task='multiclass')
        self.confmat_player = torchmetrics.ConfusionMatrix(num_classes = 6, task='multiclass')
        
        self.confusion_matrix_manufacturer = np.zeros((7, 7))
        self.confusion_matrix_guitar_type = np.zeros((4,4))
        self.confusion_matrix_pickup = np.zeros((2,2))
        self.confusion_matrix_pickup_position = np.zeros((3,3))
        self.confusion_matrix_strumming = np.zeros((2,2))
        self.confusion_matrix_player = np.zeros((6,6))

        self.test_loss = []
        self.test_acc_manufacturer = []
        self.test_acc_guitar = []
        self.test_acc_pickup = []
        self.test_acc_pickup_position = []
        self.test_acc_strumming = []
        self.test_acc_player = []
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        #x = self.maxpool(x)
        x = self.relu(self.conv2(x))
        #x = self.maxpool(x)
        x = self.relu(self.conv3(x))
        #x = self.maxpool(x)
        #x = self.relu(self.conv4(x))
        x = self.maxpool(x)

        x = x.view(-1, 64 * 61 * 91)        
        x = self.relu(self.fc1(x))
        x = x.view(-1, 128)  
        x = self.fc2(x)
        return { 
            'manufacturer': self.softmax(self.manufacturer(x)),
            'guitar_type': self.softmax(self.guitar_type(x)),
            'pickup':self.softmax(self.pickup(x)),
            'pickup_position': self.softmax(self.pickup_position(x)),
            'strumming': self.softmax(self.strumming(x)),
            'player': self.softmax(self.player(x))
        }

    def multilabel_loss_fn(self, y_hats, ys):
        losses=0
        for i, key in enumerate(y_hats):
            losses += self.criterion(y_hats[key], ys[key])
        return losses

    def multilabel_predictions(self, y_hats, ys):
        predictions=[]
        for i, key in enumerate(y_hats):
            _, predicted=torch.max(y_hats[key], 1)
            predictions.append(predicted)
        return predictions

    def multilabel_accuracy(self, predictions, ys):
        accuracies=[]
        ys = list(ys.values()) #Dictionary to list
        for idx, predicted in enumerate(predictions):
            accuracy = (predicted == ys[idx]).sum().item() / len(predicted)
            accuracies.append(accuracy)
        return accuracies

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        
        loss = self.multilabel_loss_fn(y_hat, y)
        #self.log('train_loss', loss, prog_bar=True, on_epoch=True, on_step = False)
        
        predictions = self.multilabel_predictions(y_hat, y)
        accuracies = self.multilabel_accuracy(predictions, y)
        epoch = self.trainer.current_epoch
        wandb.log({'train_loss':loss/6, 'epoch':epoch})

        return loss/6


    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        
        loss = self.multilabel_loss_fn(y_hat, y)
        #self.log('val_loss', loss/6, prog_bar=True, on_epoch=True, on_step = False)
        
        predictions = self.multilabel_predictions(y_hat, y)
        accuracies = self.multilabel_accuracy(predictions, y)
        epoch = self.trainer.current_epoch
        #self.log('val_acc_manufacturer', accuracies[0], prog_bar=True, on_epoch=True, on_step = False)
        #self.log('val_acc_guitar_type', accuracies[1], prog_bar=True, on_epoch=True, on_step = False)
        # self.log('val_acc_pick', accuracies[2], prog_bar=True, on_epoch=True, on_step = False)
        # self.log('val_acc_pickup_position', accuracies[3], prog_bar=True, on_epoch=True, on_step = False)
        # self.log('val_acc_strumming', accuracies[4], prog_bar=True, on_epoch=True, on_step = False)
        # self.log('val_acc_player', accuracies[5], prog_bar=True, on_epoch=True, on_step = False)
        wandb.log({'val_loss':loss/6, 
                   'epoch':epoch
                   })
        

        return loss/6

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        
        loss = self.multilabel_loss_fn(y_hat, y)
        #self.log('test_loss', loss/6, prog_bar=True, on_epoch=True, on_step = False)
        
        predictions = self.multilabel_predictions(y_hat, y)
        accuracies = self.multilabel_accuracy(predictions, y)
        #epoch = self.trainer.current_epoch
        # self.log('test_acc_manufacturer', accuracies[0], prog_bar=True, on_epoch=True, on_step = False)
        # self.log('test_acc_guitar_type', accuracies[1], prog_bar=True, on_epoch=True, on_step = False)
        # self.log('test_acc_pick', accuracies[2], prog_bar=True, on_epoch=True, on_step = False)
        # self.log('test_acc_pickup_position', accuracies[3], prog_bar=True, on_epoch=True, on_step = False)
        # self.log('test_acc_strumming', accuracies[4], prog_bar=True, on_epoch=True, on_step = False)
        # self.log('test_acc_player', accuracies[5], prog_bar=True, on_epoch=True, on_step = False)

        
        self.test_loss.append((loss/6).detach().cpu())
        self.test_acc_manufacturer.append(accuracies[0])
        self.test_acc_guitar.append(accuracies[1])
        self.test_acc_pickup.append(accuracies[2])
        self.test_acc_pickup_position.append(accuracies[3])
        self.test_acc_strumming.append(accuracies[4])
        self.test_acc_player.append(accuracies[5])

        #Calculate confusion matrix
        confusion_matrix_manufacturer = self.confmat_manufacturer(predictions[0], y['manufacturer'])       
        confusion_matrix_guitar_type = self.confmat_guitar_type(predictions[1], y['guitar_type'])
        confusion_matrix_pickup = self.confmat_pickup(predictions[2], y['pickup'])
        confusion_matrix_pickup_position = self.confmat_pickup_position(predictions[3], y['pickup_position'])
        confusion_matrix_strumming = self.confmat_strumming(predictions[4], y['strumming'])     
        confusion_matrix_player = self.confmat_player(predictions[5], y['player'])         

        self.confusion_matrix_manufacturer += confusion_matrix_manufacturer.cpu().numpy()
        self.confusion_matrix_guitar_type += confusion_matrix_guitar_type.cpu().numpy()
        self.confusion_matrix_pickup += confusion_matrix_pickup.cpu().numpy()
        self.confusion_matrix_pickup_position += confusion_matrix_pickup_position.cpu().numpy()
        self.confusion_matrix_strumming += confusion_matrix_strumming.cpu().numpy()
        self.confusion_matrix_player += confusion_matrix_player.cpu().numpy()
        
    def on_test_epoch_end(self):
        #Plot mean to wandb
        wandb.log({'test_loss':np.mean(self.test_loss), 
                   'test_acc_manufacturer': np.mean(self.test_acc_manufacturer),
                   'test_acc_guitar_type': np.mean(self.test_acc_guitar),
                   'test_acc_pickup': np.mean(self.test_acc_pickup),
                   'test_acc_pickup_position': np.mean(self.test_acc_pickup_position),
                   'test_acc_strumming': np.mean(self.test_acc_strumming),
                   'test_acc_player': np.mean(self.test_acc_player),
                   })

        self.test_loss = []
        self.test_acc_manufacturer = []
        self.test_acc_guitar = []
        self.test_acc_pickup = []
        self.test_acc_pickup_position = []
        self.test_acc_strumming = []
        self.test_acc_player = []

        plot_confusion_matrix(self.confusion_matrix_manufacturer, utils.get_manufacturer_labels(), filename='confusion_matrix_manufacturer.png')
        plot_confusion_matrix(self.confusion_matrix_guitar_type, utils.get_guitar_type_labels(), filename='confusion_matrix_guitar_type.png')
        plot_confusion_matrix(self.confusion_matrix_pickup, utils.get_pickup_labels(), filename='confusion_matrix_pickup.png')
        plot_confusion_matrix(self.confusion_matrix_pickup_position, utils.get_pickup_position_labels(), filename='confusion_matrix_pickup_position.png')
        plot_confusion_matrix(self.confusion_matrix_strumming, utils.get_strumming_labels(), filename='confusion_matrix_strumming.png')
        plot_confusion_matrix(self.confusion_matrix_player, utils.get_player_labels(), filename='confusion_matrix_player.png')

    ####### Optimizer
    def optimizer_selection(self):
        if self.optimizer_str == 'Adam':
            optimizer= optim.Adam(self.parameters(), lr=self.lr)
        elif self.optimizer_str == 'SGD':
            optimizer= optim.SGD(self.parameters(), lr=self.lr) 
        elif self.optimizer_str == 'Adadelta':
            optimizer= optim.Adadelta(self.parameters(), lr=self.lr)    
        elif self.optimizer_str == 'Adagrad':
            optimizer= optim.Adagrad(self.parameters(), lr=self.lr)
        elif self.optimizer_str == 'AdamW':
            optimizer= optim.AdamW(self.parameters(), lr=self.lr)
        elif self.optimizer_str == 'SparseAdam':
            optimizer= optim.SparseAdam(self.parameters(), lr=self.lr)
        elif self.optimizer_str == 'ASGD':
            optimizer= optim.ASGD(self.parameters(), lr=self.lr)  
        elif self.optimizer_str == 'LBFGS':
            optimizer= optim.LBFGS(self.parameters(), lr=self.lr)  
        elif self.optimizer_str == 'NAdam':
            optimizer= optim.NAdam(self.parameters(), lr=self.lr)  
        elif self.optimizer_str == 'RAdam':
            optimizer= optim.RAdam(self.parameters(), lr=self.lr)  
        elif self.optimizer_str == 'RMSprop':
            optimizer= optim.RMSprop(self.parameters(), lr=self.lr)  
        elif self.optimizer_str == 'Rprop':
            optimizer= optim.Rprop(self.parameters(), lr=self.lr)   
        return optimizer
    
    def configure_optimizers(self):
        optimizer =  self.optimizer_selection()
        return optimizer
  
