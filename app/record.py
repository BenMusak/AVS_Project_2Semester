import pyaudio
import wave
from scipy.io.wavfile import read
import numpy as np
import librosa

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 2
WAVE_FILENAME = "record/output.wav" # path to .wav file
DEVICE_NAME = 'Analogue 1 + 2 (Focusrite USB A'

# to find input of SOUNDCARD
# 'Analogue 1 + 2 (Focusrite USB A'
def find_input_device(pyAud):
    foundDevice = False
    dev_index = -1

    for i in range(pyAud.get_device_count()):
        dev = pyAud.get_device_info_by_index(i)
        #print((i, dev['name'], dev['maxInputChannels']))
    
        if dev['name'] == DEVICE_NAME:
            foundDevice = True
            dev_index = i 

    return foundDevice, dev_index

# creating .wav file
def create_wav(pyAud, frames):
    wf = wave.open(WAVE_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(pyAud.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    
    pass

def record_audio():
    pyAud = pyaudio.PyAudio()
    
    foundDevice = True
    foundDevice, dev_index = find_input_device(pyAud)
    
    if not foundDevice:
        print('Cannot record audio')
        return None
    
    stream = pyAud.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK,
                input_device_index=dev_index)
    
    print("* recording")

    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("* done recording")

    stream.stop_stream()
    stream.close()
    pyAud.terminate()
    
    create_wav(pyAud, frames)

    #signal, sr = librosa.load(WAVE_FILENAME)
    #aud_data = read(WAVE_FILENAME)
    #aud_data = np.array(aud_data[1],dtype=float)

    #return signal, sr
