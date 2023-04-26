# convert audio file '.wav' to 
import librosa
import numpy as np
import joblib

WAVE_PATH = "record/output.wav"
MODEL_PATH = "models/KNN_model.sav"

def load_file():
    try:   
        signal, sr = librosa.load(WAVE_PATH)
        return signal, sr
    except:
        print("File not found.")
        return None, None

def convert(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    delta_mfcc = librosa.feature.delta(mfcc, order=1)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    
    all_mfcc = np.concatenate([mfcc, delta_mfcc, delta2_mfcc])
    features = np.sum(all_mfcc, axis=1)
    
    
    features = features.reshape(1, -1)
    print(features.shape)

    return features

def load_model():
    return joblib.load(MODEL_PATH)

y, sr = load_file()
features = convert(y, sr)