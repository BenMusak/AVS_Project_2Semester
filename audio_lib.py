import librosa
import os
import time
import numpy as np
import IPython.display as ipd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
            

def reshape_data(X):
    #print("Reshaping the dataset...")
    #X = np.asarray(X)
    X_reshaped = X.reshape(X.shape[0] * X.shape[1])
    return X_reshaped


# Function to load all the sound files from the directories
def load_files_from_directories(directories):

    # Loop over the directories and get a list of all the sound files in each directory
    sound_files = [[] for i in range(len(directories))]
    for idx, directory in enumerate (directories):
        sound_files[idx] += [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.wav')]

    print("Directories found: " + str(len(sound_files)))
    print("Sound files found in first directory: " + str(len(sound_files[0])))

    # Loop over the sound files and load them using librosa.load
    strum_list = [[] for i in range(len(directories))]
    for idx, strum_file in enumerate(sound_files):
        # Load the sound file using librosa.load
        print("Loading files from directory: " + str(idx + 1) + " of " + str(len(directories)))
        for i in range(len(strum_file)):
            audio_file = librosa.load(strum_file[i], sr=44100, duration=2)
            normalized = np.array(librosa.util.normalize(audio_file[0]))
            strum_list[idx].append((normalized, audio_file[1]))
            #strum_list[idx].append(librosa.load(strum_file[i], sr=48000, duration=2))
        print("Files loaded from directory: " + str(idx + 1) + " of " + str(len(directories)) + " - " + str(len(strum_file)) + " files loaded")

    return strum_list


def extract_mel_mfcc_multible_files(sound_files, label, display=False):
    mfccs = []
    mels = []

    # Loop over the sound files and load them using librosa.load
    print("Extracting MFCCs and Mel Spectrograms...")
    for sound_file in sound_files:

        # Load the sound file using librosa.load
        mfcc = extract_MFCCs(sound_file[0], sound_file[1], res=13)
        delta_mfcc = extract_delta_MFCCs(mfcc, order=1)
        delta2_mfcc = extract_delta_MFCCs(mfcc, order=2)
        comprehensive_mfccs = np.concatenate([mfcc, delta_mfcc, delta2_mfcc])

        comprehensive_mfccs = np.sum(comprehensive_mfccs, axis=1)
        mel = extract_Mel(sound_file[0], sound_file[1])
        
        # Visualize the MFCCs and Mel Spectrograms if display is True, else sum the MFCCs
        if display:
            visualize_MFCCs_Mel(np.asanyarray(mfcc), np.asanyarray(mel), sound_file[1])
        else:
            mfcc = np.sum(mfcc, axis=1)
            

        mfccs.append(comprehensive_mfccs)
        mels.append(mel)

    # Print the shape of the MFCCs and Mel Spectrograms
    print("MFCCs shape for label {}: {}".format(label, np.asarray(mfccs).shape))
    return mels, mfccs


def extract_mel_mfcc_multible_files_no_sum(sound_files, label, display=False):
    mfccs = []
    mels = []

    # Loop over the sound files and load them using librosa.load
    print("Extracting MFCCs and Mel Spectrograms...")
    print("Number of files: " + str(len(sound_files)))
    print("label: " + str(label))
    for sound_file in sound_files:

        # Load the sound file using librosa.load
        mfcc = extract_MFCCs(sound_file[0], sound_file[1], res=13)
        delta_mfcc = extract_delta_MFCCs(mfcc, order=1)
        delta2_mfcc = extract_delta_MFCCs(mfcc, order=2)
        comprehensive_mfccs = np.concatenate([mfcc, delta_mfcc, delta2_mfcc])

        #comprehensive_mfccs = np.sum(comprehensive_mfccs, axis=1)
        mel = extract_Mel(sound_file[0], sound_file[1])
        
        # Visualize the MFCCs and Mel Spectrograms if display is True, else sum the MFCCs
        if display:
            visualize_MFCCs_Mel(np.asanyarray(mfcc), np.asanyarray(mel), sound_file[1])

        #print(comprehensive_mfccs.shape)    
        
        mfccs_reshaped = reshape_data(comprehensive_mfccs)

        if mfccs_reshaped.shape[0] != 6747:
            continue

        mfccs.append(mfccs_reshaped)
        mels.append(mel)

    # Print the shape of the MFCCs and Mel Spectrograms
    print("MFCCs shape for label {}: {}".format(label, np.asarray(mfccs).shape))
    return mels, mfccs


# Function to load a single audio file
def load_audio_file(file_path):
    print("Loading Audio File...")
    signal, sr = librosa.load(file_path)
    return signal, sr


def play_audio_file(y, sr):
    ipd.Audio(y, rate=sr)


def get_samples(signal):
    print(signal.shape)
    return signal.shape[0]


def extract_MFCCs(y, sr, res=11):
    MFCCs = librosa.feature.mfcc(y=y, n_mfcc=res, sr=sr)
    return MFCCs


def extract_delta_MFCCs(y, order):
    delta_MFCCs = librosa.feature.delta(y, order=order)
    return delta_MFCCs


def extract_Mel(y, sr, res=128):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=res)
    return S


def mean_mfccs(x):
    return [np.mean(feature) for feature in librosa.feature.mfcc(x)]


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


def dataset_combine_multible_files(MFCCs, Mels, sr, label, display=False):

    # Give the strums a label
    labeled_parts = []
    for i in range(0, len(MFCCs)):
        labeled_parts.append((MFCCs[i], label))

    # Visualize the non-silent parts if display is True
    if display:
        for i in range(0, len(MFCCs)):
            visualize_MFCCs_Mel(MFCCs[i], Mels[i], sr)

    # Split the labeled parts into training and test sets
    train, test = train_test_split(labeled_parts, test_size=0.2)

    # Split the training set into training and validation sets
    #train, val = train_test_split(train, test_size=0.2)

    # Extract the MFCCs and labels from the training, validation, and test sets
    train_x, train_y = zip(*train)
    #val_x, val_y = zip(*val)
    test_x, test_y = zip(*test)

    # Convert the MFCCs to numpy arrays
    train_x = np.array(train_x)
    #val_x = np.array(val_x)
    test_x = np.array(test_x)

    # Convert the labels to numpy arrays
    train_y = np.array(train_y)
    #val_y = np.array(val_y)
    test_y = np.array(test_y)

    return train_x, train_y, test_x, test_y


def dataset_split(MFCCs, Mel, sr, label, res=100, diff_extremes=43, display=False):

    print("Splitting dataset...")

    print("Finding non-silent parts...")
    # Find places in the MFCCs where the audio is silent looking at the first coefficient
    non_silent_parts = np.where(MFCCs[0, :] > -500)[0]
    print(non_silent_parts)

    # For Each new non-silent part after a silent part, merge the next 50 non-silent parts
    # into the current non-silent part. Then remove the next non-silent parts until a new silent part is meet.
    # Remember to check for out of bounds
    i = 0
    strum_single = []
    strums_list = []
    new_strum = True
    check_index = []
    while i < len(non_silent_parts) - 1:
        diff = abs(MFCCs[0, non_silent_parts[i]] - MFCCs[0, non_silent_parts[i - 1]])
        #print(diff)
        if diff > diff_extremes and new_strum:
            if non_silent_parts[i] + 1 == non_silent_parts[i + 1]:
                for j in range(0, res):
                    strum_single.append(MFCCs[:, non_silent_parts[i + j]])
                    check_index.append(non_silent_parts[i + j])
                    try:
                        if MFCCs[:, i + j].all() == MFCCs[:, non_silent_parts[i]].all():
                            non_silent_parts = np.delete(non_silent_parts, np.arange(i, i))
                    except IndexError:
                        pass
                strums_list.append(strum_single)
                strum_single = []
                new_strum = False
        else:
            new_strum = True
        i += 1

    # Convert the list of strums to a numpy array
    strums_list = np.asarray(strums_list)

    # Sum the MFCCs of each strum to get a single value for each strum
    strums_list = np.sum(strums_list, axis=1)

    print(strums_list.shape)

    # Give the strums a label
    labeled_parts = []
    for i in range(0, len(strums_list)):
        labeled_parts.append((strums_list[i], label))


    #print("Non-silent parts:" + str(len(non_silent_parts)))
    print("Strums:" + str(len(strums_list)))
    #print("One strum:" + str(strums_list[0]))
    print("All parts:" + str(len(MFCCs[1])))

    # Visualize the non-silent parts
    if display:
        visualize_MFCCs_Mel(MFCCs[:, check_index], Mel[:, check_index], sr)

    # Split the labeled parts into training and test sets
    train, test = train_test_split(labeled_parts, test_size=0.2)

    # Split the training set into training and validation sets
    #train, val = train_test_split(train, test_size=0.2)

    # Extract the MFCCs and labels from the training, validation, and test sets
    train_x, train_y = zip(*train)
    #val_x, val_y = zip(*val)
    test_x, test_y = zip(*test)

    # Convert the MFCCs to numpy arrays
    train_x = np.array(train_x)
    #val_x = np.array(val_x)
    test_x = np.array(test_x)

    # Convert the labels to numpy arrays
    train_y = np.array(train_y)
    #val_y = np.array(val_y)
    test_y = np.array(test_y)

    return train_x, train_y, test_x, test_y


def save_dataset(train_x, train_y, test_x, test_y):
    print("Saving dataset...")
    np.savez("dataset.npz", train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y)


def load_dataset(dataset_path):
    # Load the numpy dataset
    dataset = np.load(dataset_path)
    # Split the dataset into features and targets
    print(dataset.files)
    train_x = np.absolute(dataset['train_x'])
    train_y = np.absolute(dataset['train_y'])
    test_x = np.absolute(dataset['test_x'])
    test_y = np.absolute(dataset['test_y'])

    return train_x, train_y, test_x, test_y
