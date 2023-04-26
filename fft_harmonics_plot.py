import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
from scipy.fft import fft,  fftfreq

#HYPERPARAMETER
LP_DIR = "LP_E2.wav"
SC_DIR = "SC_E2.wav"
FIRST_HARMONIC = 82.41 #Hz

def read_audio(PATH):
    print("Reading audio file: ", PATH)
    samplerate, signal = wavfile.read(LP_DIR)

    #Normalize the signal
    signal = np.array([(signal / np.max(np.abs(signal))) * 32767], np.int16)
    signal = np.squeeze(signal)
    #There are 2 channels in the wav signal. We only need 1 channel
    signal = signal.sum(axis=1) / 2

    samples = signal.shape[0]
    seconds = samples/samplerate

    print(f"Shape: {signal.shape} | Samplerate: {samplerate} | Samples: {samples} | Seconds: {seconds}")
    return signal, samplerate, samples

def do_fft(signal, samplerate, samples):

    Ts = 1.0/samplerate # Sampingrate in seconds
    print("Ts (Samplingrate in seconds): ", Ts)

    FFT = abs(fft(signal))

    freq_bins = fftfreq(samples, Ts) #Each frequency bin is spaced 
    print("Freq Bins: ", freq_bins)

    #The FFT is symmetric at the middle, so we can remove the last half
    FFT_side = FFT[range(samples//2)]
    freq_bins_side = freq_bins[range(samples//2)]

    #Since we remove the latter half, we have to multiply all the values by 2
    FFT_side = FFT_side * 2
    #And normalize using the amount of samples
    FFT_side = FFT_side/samples

    #Convert to dB
    ref = 1 #reference value used for dBFS scale. 32768 for int16 and 1 for float
    FFT_side = 20 * np.log10(FFT_side)

    #We don't need to show all of the bins, as a lot of them are very low, so we split the array up (should show up to 1k Hz)
    freq_bins_side = freq_bins_side[:8000]
    FFT_side = FFT_side[:8000]

    return FFT_side, freq_bins_side

def calculate_harmonics(fundamental_freq, amount):
    return np.arange(fundamental_freq, fundamental_freq*amount, fundamental_freq)

def plot_signal(x, y, vlines):
    plt.figure()
    plt.grid()
    plt.plot(y, x, "b", linewidth=0.8) # plotting the positive fft spectrum
    plt.vlines(vlines, min(x), max(x), linestyles='dashed', colors='red', linewidth=0.8)
    plt.ylim(min(x), max(x))
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude [dB]')
    plt.show()



def main():
    #Load signal
    sig, fs, N = read_audio(SC_DIR)

    #Calculate FFT and Bins
    signal, bins = do_fft(sig, fs, N)

    #Calculate the harmonics (just for plotting the vertical lines)
    harmonics = calculate_harmonics(FIRST_HARMONIC, 13)
    print("harmonics", harmonics)
    #Manually type the harmonics, as the automated one does not fit 100% with the audio signal
    harmonics = [82.41, 164.82, 247.3, 329.7, 412.1, 496, 580, 666, 753, 835, 920, 1005]

    #Plot signal
    plot_signal(signal, bins, harmonics)


if __name__ == "__main__":
    main()



