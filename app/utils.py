import librosa
import matplotlib.pyplot as plt
import numpy as np
import joblib
from LDA import LDA
from sklearn.preprocessing import StandardScaler


def load_file(wav_path):
    try: 
        audio_file, sr = librosa.load(wav_path, sr=48000, duration=2)
        normalized = np.array(librosa.util.normalize(audio_file))
        return normalized, sr
    except:
        print("File not found.")
        return None, None


def reshape_data(X):
    #print("Reshaping the dataset...")
    #X = np.asarray(X)
    X_reshaped = X.reshape(X.shape[0] * X.shape[1])
    return X_reshaped


def convert(y, sr, scaler_path):

    mfccs = []

    #trimmed = librosa.util.fix_length(data=y, size=int(sr * 5))

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    delta_mfcc = librosa.feature.delta(mfcc, order=1)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)

    fig = visualize_MFCCs_Mel(mfcc, librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000), sr)

    all_mfcc = np.concatenate([mfcc, delta_mfcc, delta2_mfcc])
    features = reshape_data(all_mfcc)

    mfccs.append(features)
    mfccs = np.asarray(mfccs)

    print(mfccs.shape)
    
    # Scale the data to be between -1 and 1
    scaler = joblib.load(scaler_path)

    # Transform the training and testing data
    x_test_scaled = scaler.transform(mfccs)

    print("New shape: ", x_test_scaled.shape)

    return x_test_scaled, mfccs, fig


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


def load_model(model_path_KNN, model_path_LDA, model_path_SVM):
    model_KNN = joblib.load(model_path_KNN)
    model_LDA = joblib.load(model_path_LDA)
    mode_SVM = joblib.load(model_path_SVM)
    return model_KNN, model_LDA, mode_SVM

#y, sr = load_file(WAVE_PATH)
#features = convert(y, sr)