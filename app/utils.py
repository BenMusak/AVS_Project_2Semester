import librosa
import matplotlib.pyplot as plt
import numpy as np
import joblib
from LDA import LDA


def load_file(wav_path):
    try:   
        signal, sr = librosa.load(wav_path)
        normalized = np.array(librosa.util.normalize(signal))
        return normalized, sr
    except:
        print("File not found.")
        return None, None


def convert(y, sr):

    trimmed = librosa.util.fix_length(data=y, size=int(sr * 2))

    mfcc = librosa.feature.mfcc(y=trimmed, sr=sr, n_mfcc=13)
    delta_mfcc = librosa.feature.delta(mfcc, order=1)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)

    visualize_MFCCs_Mel(mfcc, librosa.feature.melspectrogram(y=trimmed, sr=sr, n_mels=128, fmax=8000), sr)

    all_mfcc = np.concatenate([mfcc, delta_mfcc, delta2_mfcc])
    features = np.sum(all_mfcc, axis=1)
    
    #features = features.reshape(1, -1)
    #print(features.shape)

    return features


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
    plt.show()


def load_model(model_path_KNN, model_path_LDA):
    model_KNN = joblib.load(model_path_KNN)
    model_LDA = joblib.load(model_path_LDA)
    return model_KNN, model_LDA

#y, sr = load_file(WAVE_PATH)
#features = convert(y, sr)