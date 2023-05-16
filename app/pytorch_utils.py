import torch
import torch.nn as nn
import pytorch_lightning as pl

class CNNClassifier(pl.LightningModule):
    def __init__(self):
        super(CNNClassifier, self).__init__()

        #In channels, Out channels
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=0)  
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)

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
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
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

    def multilabel_accuracy(self, predictions, ys):
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
    return ['JM', 'VS', 'BH', 'JG', 'KB', 'JH']


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

def multilabel_predictions(y_hats):
    predictions=[]
    for i, key in enumerate(y_hats):
        _, predicted=torch.max(y_hats[key], 1)
        predictions.append(predicted)
    return predictions
