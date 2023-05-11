import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
from audio_lib import extract_MFCCs, extract_delta_MFCCs, reshape_data

def load_data_from_csv(data_dir, csv_dir): #Label can be 'guitar', 'pickup', 'player' etc.
    print("Loading files from: ", data_dir)
    df = pd.read_csv(csv_dir)
    data = []
    labels = []
    names = []

    for idx in tqdm(range(df.shape[0])):
        try:
            # Load sound file

            audio_path = os.path.join(data_dir, df.iloc[idx]['name'][:-3]+'wav') #The csv file has .png instead of .wav so we do this.
            audio = librosa.load(audio_path, sr=44100, duration=2) #Load
            normalized = np.array(librosa.util.normalize(audio[0])) #Normalize
            # Get label
            row = df.iloc[idx]
            manufacturer = manufacturer_str_to_int(row['manufacturer'])
            guitar_type = guitar_type_str_to_int(row['guitar_type'])
            pickup =pickup_str_to_int(row['pickup'])
            pickup_position = pickup_position_str_to_int(row['pickup_position'])
            strumming= strumming_str_to_int(row['strumming'])
            player = player_str_to_int(row['player'])
            #Append
            data.append(normalized)
            labels.append({'manufacturer': manufacturer,
                            'guitar_type': guitar_type,
                            'pickup': pickup,
                            'pickup_position': pickup_position,
                            'strumming' : strumming,
                            'player': player    
                            })
            names.append(row['name']) #For debugging
        except:
            print("Could not find ", audio_path, ". Skipping")

    return data, labels, names

def extract_mfccs(soundfiles, sr=44100):
    print("Extracting MFCCs")
    mfccs = []
    for idx in tqdm(range(len(soundfiles))):
        mfcc = extract_MFCCs(soundfiles[idx], sr, res=13)
        delta_mfcc = extract_delta_MFCCs(mfcc, order=1)
        delta2_mfcc = extract_delta_MFCCs(mfcc, order=2)
        comprehensive_mfccs = np.concatenate([mfcc, delta_mfcc, delta2_mfcc])

        comprehensive_mfccs = np.sum(comprehensive_mfccs, axis=1)
        #mfccs_reshaped = reshape_data(comprehensive_mfccs)

        mfccs.append(comprehensive_mfccs)

    return mfccs    

def list_get_desired_label(list_, label):
    print("Getting desired label: ", label)
    for idx in tqdm(range(len(list_))):
        list_[idx] = list_[idx][label]
    return list_


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

