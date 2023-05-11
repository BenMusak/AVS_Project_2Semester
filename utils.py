import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues, filename=None):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)

    plt.figure(figsize=(6, 7))
    
    im = plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar(im, fraction=0.046, pad=0.04)

    plt.xticks(np.arange(len(classes)), classes, rotation=45)
    plt.yticks(np.arange(len(classes)), classes)
    
    plt.title(title)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    
    for i in range(len(classes)):
        for j in range(len(classes)):
            plt.text(j, i, cm[i, j], horizontalalignment='center', color='white' if cm[i, j] > cm.max() / 2 else 'black', fontsize=14)
    
    if filename:
        plt.savefig(filename)
    

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
    return ['JM', 'VS', 'BH', 'JG', 'KB','AL']


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

