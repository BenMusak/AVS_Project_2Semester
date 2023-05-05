import pyaudio
import wave
from scipy.io.wavfile import read

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 5
DEVICE_NAME = 'Analogue 1 + 2 (Focusrite USB A'

# to find input of SOUNDCARD
# 'Analogue 1 + 2 (Focusrite USB A'
def find_input_device(pyAud):
    foundDevice = False
    dev_index = -1

    for i in range(pyAud.get_device_count()):
        dev = pyAud.get_device_info_by_index(i)
        print((i, dev['name'], dev['maxInputChannels']))
    
        if dev['name'] == DEVICE_NAME:
            foundDevice = True
            dev_index = i 

    return foundDevice, dev_index

# creating .wav file
def create_wav(pyAud, frames, output_path):
    wf = wave.open(output_path, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(pyAud.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    
    pass

def record_audio(output_path):
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
    
    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    pyAud.terminate()
    
    create_wav(pyAud, frames, output_path)
