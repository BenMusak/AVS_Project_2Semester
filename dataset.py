import librosa
import KNN as knn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



def load_data(data_path, label_path):
    print("Loading the dataset...")
    X = np.load(data_path) # data
    y = np.load(label_path) # label

    return X, y


def reshape_data(X):
    print("Reshaping the dataset...")
    X = np.asarray(X)
    X_reshaped = X.reshape(-1, X.shape[1] * X.shape[2])
    return X_reshaped


def split_dataset(X, y, test_split):
    print("Splitting the dataset...")
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split, random_state=42, shuffle=True)
    
    labed_parts = list(zip(X, y))

    # Split the labeled parts into training and test sets
    train, test = train_test_split(labed_parts, test_size=test_split)

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


def scale_data(X_train, X_test):
    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled


def extract_delta_MFCCs(y, order):
    delta_MFCCs = librosa.feature.delta(y, order=order)
    return delta_MFCCs


def convert_to_mfcc(data):
    mfccs = []
    print("Converting to MFCCs...")
    for y in data:
        converted = librosa.db_to_power(y)
        mfcc = librosa.feature.mfcc(S=converted)
        #visualize_MFCCs_Mel(mfcc, y, 44100)
        mfcc_delta = extract_delta_MFCCs(mfcc, 1)
        mfcc_delta2 = extract_delta_MFCCs(mfcc, 2)
        comprehensive_mfccs = np.concatenate([mfcc, mfcc_delta, mfcc_delta2])
        #comprehensive_mfccs = np.sum(comprehensive_mfccs, axis=1)
        mfccs.append(comprehensive_mfccs)
    return mfccs


def visualize_MFCCs_Mel(MFCCs, Mel, sr):
    print("Visualizing MFCCs...")
    fig, ax = plt.subplots(nrows=2, sharex=True)
    img_mel = librosa.display.specshow(Mel,
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


def create_dataset(data_path, label_path, test_split=0.2):
    X, y = load_data(data_path, label_path)
    mfccs = convert_to_mfcc(X)
    reshaped = reshape_data(mfccs)
    X_train, X_test, y_train, y_test = split_dataset(reshaped, y, test_split)

    return X_train, X_test, y_train, y_test