import torch
import torch.nn as nn
import pytorch_lightning as pl

class CNNClassifier(pl.LightningModule):
    def __init__(self):
        super(CNNClassifier, self).__init__()

        #In channels, Out channels
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        
        self.fc1 = nn.Linear(128 * 8 * 11, 256)

        self.manufacturer = nn.Linear(256, 7)
        self.guitar_type = nn.Linear(256, 4)
        self.pickup = nn.Linear(256, 2)
        self.pickup_position = nn.Linear(256, 3)
        self.strumming = nn.Linear(256, 2)
        self.player = nn.Linear(256, 5)
        
        self.relu = nn.ReLU()
        
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

        return { 
            'manufacturer': self.manufacturer(x),
            'guitar_type': self.guitar_type(x),
            'pickup':self.pickup(x),
            'pickup_position': self.pickup_position(x),
            'strumming': self.strumming(x),
            'player': self.player(x)
        }


def multilabel_predictions(y_hats):
    predictions=[]
    for i, key in enumerate(y_hats):
        _, predicted=torch.max(y_hats[key], 1)
        predictions.append(predicted)
    return predictions

def multilabel_accuracy(predictions, ys):
    accuracies=[]
    ys = list(ys.values()) #Dictionary to list
    for idx, predicted in enumerate(predictions):
        accuracy = (predicted == ys[idx]).sum().item() / len(predicted)
        accuracies.append(accuracy)
    return accuracies

#########Labels########
def get_manufacturer_labels():
    return ['Gibson', 'Epiphone', 'Supreme', 'Axtech', 'Fender', 'Hansen', 'Squier']
    
def get_guitar_type_labels():
    return ['LP', 'SG', 'SC', 'TC']

def get_pickup_labels():
    return ['Humbucker','Single Coil']   

def get_pickup_position_labels():
    return ['Bridge', 'Middle', 'Neck']
    
def get_strumming_labels():
    return ['Open','Amajor']

def get_player_labels():
    return ['JM', 'VS', 'BH', 'JG', 'KB']


#######Labels -> Int####
def manufacturer_str_to_int(label):
    return get_manufacturer_labels().index(label)

def guitar_type_str_to_int(label):
    return get_guitar_type_labels().index(label)

def pickup_str_to_int(label):
    return get_pickup_labels().index(label)

def pickup_position_str_to_int(label):
    return get_pickup_position_labels().index(label)

def strumming_str_to_int(label):
    return get_strumming_labels().index(label)

def player_str_to_int(label):
    return get_player_labels().index(label)

###### Convert
# TODO : why do we need this? 
def convert_mel_spec_t(mel_spec):
    mel_spec_t = torch.tensor(mel_spec)
    mel_spec_t = torch.unsqueeze(mel_spec_t, dim=0)
    return mel_spec_t


