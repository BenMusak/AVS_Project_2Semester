import librosa
import numpy as np
import joblib


def load_file(wav_path):
    try:   
        signal, sr = librosa.load(wav_path)
        return signal, sr
    except:
        print("File not found.")
        return None, None

def convert(y, sr):

    trimmed = librosa.util.fix_length(data=y, size=int(sr * 2))

    mfcc = librosa.feature.mfcc(y=trimmed, sr=sr, n_mfcc=13)
    delta_mfcc = librosa.feature.delta(mfcc, order=1)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    
    print(len(mfcc))

    all_mfcc = np.concatenate([mfcc, delta_mfcc, delta2_mfcc])
    features = np.sum(all_mfcc, axis=1)
    
    features = features.reshape(1, -1)
    print(features.shape)

    return features

def load_model(model_path):
    return joblib.load(model_path)

#y, sr = load_file(WAVE_PATH)
#features = convert(y, sr)