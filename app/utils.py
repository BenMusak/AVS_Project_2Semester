import librosa
import matplotlib.pyplot as plt
import numpy as np
import joblib
from pytorch_utils import CNNClassifier
import torch
from PIL import Image
from torchvision.io import read_image, ImageReadMode
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

def load_file(wav_path, sr):
    try: 
        audio_file, sr = librosa.load(wav_path, sr=sr, duration=2)
        normalized = np.array(librosa.util.normalize(audio_file))
        return normalized, sr
    except:
        print("File not found.")
        return None, None

def reshape_data(X):
    X_reshaped = X.reshape(X.shape[0] * X.shape[1])
    return X_reshaped

def convert_mfcc(y, sr, scaler_path):

    mfccs = []

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    delta_mfcc = librosa.feature.delta(mfcc, order=1)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)

    #fig = visualize_MFCCs_Mel(mfcc, librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000), sr)

    all_mfcc = np.concatenate([mfcc, delta_mfcc, delta2_mfcc])
    features = reshape_data(all_mfcc)

    mfccs.append(features)
    mfccs = np.asarray(mfccs)
    
    # Scale the data to be between -1 and 1
    scaler = joblib.load(scaler_path)

    # Transform the training and testing data
    x_test_scaled = scaler.transform(mfccs)

    return x_test_scaled, mfccs#, fig

def convert_mel_spectrogram(y, sr):
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)

    mel_spec = Image.fromarray(mel_spec)
    mel_spec = mel_spec.convert('L')
    mel_spec.save('mel_spectogram.png')
    spectrogram = read_image('mel_spectogram.png', ImageReadMode.GRAY)

    spectrogram = spectrogram.float()
    spectrogram = spectrogram / 255 # Normalize values from [0-255] to [0-1]

    return spectrogram

def visualize_MFCCs_Mel(MFCCs, Mel, sr):
    print("Visualizing MFCCs...")
    fig, ax = plt.subplots(nrows=2, sharex=True)
    img_mel = librosa.display.specshow(librosa.power_to_db(Mel, ref=np.max),
                                        x_axis='time', y_axis='mel', fmax=8000,
                                        ax=ax[0])
    fig.colorbar(img_mel, ax=[ax[0]])
    ax[0].set(title='Mel spectrogram')
    ax[0].label_outer()
    img_MFCCs = librosa.display.specshow(MFCCs, x_axis='time', sr=sr)
    fig.colorbar(img_MFCCs, ax=[ax[1]])
    ax[1].set(title='MFCC')
    plt.title('MFCCs')
    
    return fig

def load_models(model_path_KNN, model_path_LDA, model_path_SVM, model_path_CNN):
    model_KNN = joblib.load(model_path_KNN)
    model_LDA = joblib.load(model_path_LDA)
    model_SVM = joblib.load(model_path_SVM)
    model_CNN = CNNClassifier().load_from_checkpoint(model_path_CNN, map_location=torch.device('cpu')).eval()
    return model_KNN, model_LDA, model_SVM, model_CNN


def load_model(file_path, labels):
    return joblib.load(file_path)


def load_CNN_model(file_path):
    return CNNClassifier().load_from_checkpoint(file_path, map_location=torch.device('cpu')).eval()


def get_labels(label):
    # label for dataset
    if label == 'manufacturer':
        label_manufacturer = {
            0 : 'Gibson',
            1 : 'Epiphone', 
            2 : 'Supreme',
            3 : 'Axtech',
            4 : 'Fender',
            5 : 'Hansen',
            6 : 'Squier'
            }
        return label_manufacturer

    # label for dataset
    elif label == 'guitar_type':
        label_guitar_type = {
            0 : 'LP',
            1 : 'SG', 
            2 : 'SC',
            3 : 'TC'
            }
        return label_guitar_type

    # label for dataset
    elif label == 'pickup':
        label_pickup = {
            0 : 'Humbucker',
            1 : 'Single Coil'
            }
        return label_pickup

    # label for dataset
    elif label == 'pickup_position':
        label_pickup_position = {
            0 : 'Bridge',
            1 : 'Middle',
            2 : 'Neck'
            }
        return label_pickup_position

    # label for dataset
    elif label == 'strumming':
        label_strumming = {
            0 : 'Open',
            1 : 'Amajor'
            }
        return label_strumming

    # label for dataset
    elif label == 'player':
        label_player = {
            0 : 'JM',
            1 : 'VS', 
            2 : 'BH',
            3 : 'JG',
            4 : 'KB',
            5 : 'AL',
            }
        return label_player

def get_labels_reverse(label):

    if label == 'manufacturer':
        label_manufacturer = {
            'Gibson' : 0,
            'Epiphone' : 1, 
            'Supreme' : 2,
            'Axtech' : 3,
            'Fender' : 4,
            'Hansen' : 5,
            'Squier' : 6
            }
        return label_manufacturer

    # label for dataset
    elif label == 'guitar_type':
        label_guitar_type = {
            'LP' : 0,
            'SG' : 1, 
            'SC' : 2,
            'TC' : 3
            }
        return label_guitar_type

    # label for dataset
    elif label == 'pickup':
        label_pickup = {
            'Humbucker' : 0,
            'Single Coil' : 1
            }
        return label_pickup

    # label for dataset
    elif label == 'pickup_position':
        label_pickup_position = {
            'Bridge' : 0,
            'Middle' : 1,
            'Neck' : 2
            }
        return label_pickup_position

    # label for dataset
    elif label == 'strumming':
        label_strumming = {
            'Open' : 0,
            'Amajor' : 1
            }
        return label_strumming

    # label for dataset
    elif label == 'player':
        label_player = {
            'JM' : 0,
            'VS' : 1, 
            'BH' : 2,
            'JG' : 3,
            'KB' : 4,
            'AL' : 5
            }
        return label_player


def get_reports(y_pred, y_test):
    print(f'preds : {y_pred}')
    print(f'test : {y_test}')

    print(f'type of preds: {type(y_pred[0][0])}')
    print(f'type of test: {type(y_test[0][0])}')
    # Convert str to int for each label in y_pred and y_test
    print(f'----------------')
    print(f'Are they equal length? : {len(y_pred) == len(y_test)}')
    
    #for i in range(len(y_pred)):
        #print(f'----------------------')
        #print(f'get reports : {i}')
        #y_pred = [int(j) for j in y_pred[i]]
        #y_test = [int(j) for j in y_test[i]]
    
    guitar_type_pred, guitar_type_true = [], []
    pickup_pred, pickup_true = [], []
    pickup_pos_pred, pickup_pos_true = [], []
    strumming_pred, strumming_true = [], []
    player_pred, player_true = [], []    

    for i in range(len(y_pred)):
        # guitar type - 0
        guitar_type_pred.append(y_pred[i][0])
        guitar_type_true.append(y_test[i][0])
        
        # pickup type - 1
        pickup_pred.append(y_pred[i][1])
        pickup_true.append(y_test[i][1])

        # pickup position - 2
        pickup_pos_pred.append(y_pred[i][2])
        pickup_pos_true.append(y_test[i][2])

        # strumming - 3
        strumming_pred.append(y_pred[i][3])
        strumming_true.append(y_test[i][3])

        # player - 4
        player_pred.append(y_pred[i][4])
        player_true.append(y_test[i][4])


    report_guitar_type = classification_report(guitar_type_true, guitar_type_pred, output_dict=True)
    report_pickup = classification_report(pickup_true, pickup_pred, output_dict=True)
    report_pickup_pos = classification_report(pickup_pos_true, pickup_pos_pred, output_dict=True)
    report_strumming = classification_report(strumming_true, strumming_pred, output_dict=True)
    report_play = classification_report(player_true, player_pred, output_dict=True)
    
    return report_guitar_type, report_pickup, report_pickup_pos, report_strumming, report_play


def get_predictions(y_pred):
    report_guitar_type = int(y_pred[0])
    report_pickup = int(y_pred[1])
    report_pickup_pos = int(y_pred[2])
    report_strumming = int(y_pred[3])
    report_play = int(y_pred[4])

    return report_guitar_type, report_pickup, report_pickup_pos, report_strumming, report_play