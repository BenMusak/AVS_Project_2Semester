# convert audio file '.wav' to 
import librosa
import numpy as np

WAVE_FILENAME = "record/output.wav"

def load_file():
    try:   
        signal, sr = librosa.load(WAVE_FILENAME)
        return signal, sr
    except:
        print("File not found.")
        return None, None

def convert(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr,  n_mfcc=13)
    data_mfcc = np.array(mfcc)
    
    return data_mfcc