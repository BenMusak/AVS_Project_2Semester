import pyaudio
import wave

pyAud = pyaudio.PyAudio()

# to find input of SOUNDCARD
# 'Analogue 1 + 2 (Focusrite USB A'
# 'Analogue 1 + 2 (wc4800_8211)'
DEVICE_NAME = 'Analogue 1 + 2 (Focusrite USB A'
foundUSBMic = False
dev_index = -1

for i in range(pyAud.get_device_count()):
    dev = pyAud.get_device_info_by_index(i)
    print((i, dev['name'], dev['maxInputChannels']))
    
    if dev['name'] == DEVICE_NAME:
        foundUSBMic = True
        dev_index = i 

print(foundUSBMic)
print(dev_index)