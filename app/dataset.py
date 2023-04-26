import numpy as np
from sklearn.preprocessing import StandardScaler
import audio_lib as al

class Dataset:
    def __init__(self):
        self.X_train = []
        self.X_test = []
        self.y_train = []
        self.y_test = []

    def create_dataset(self, dir_lst):
        strum_lst = al.load_files_from_directories(dir_lst)

        for label in range(len(strum_lst)):
            print("lenght of strum_list: ", len(strum_lst[label]))

            # Extract Mek and MFCC features from the current audio file
            mels, mffcs = al.extract_mel_mfcc_multible_files(strum_lst[label], label, display=False)

            # Split the data into train and test data
            train_x_, train_y_, test_x_, test_y_ = al.dataset_combine_multible_files(mffcs, mels, strum_lst[label], label)

            # Append the data to the lists so that they can be concatenated later
            self.X_train.append(train_x_)
            self.y_train.append(train_y_)
            self.X_test.append(test_x_)
            self.y_test.append(test_y_)

        self.X_train = np.concatenate(self.X_train)
        self.y_train = np.concatenate(self.y_train)
        self.X_test = np.concatenate(self.X_test)
        self.y_test = np.concatenate(self.y_test)


    def scale(self):
        scaler = StandardScaler()
        scaler.fit(self.X_train)
        
        self.X_train = scaler.transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)


##################################################
##                   test Dataset               ##
##################################################
#dirs = [
#    r'C:\Users\Kata\OneDrive - Aalborg Universitet\CE8-DESKTOP-T44IC5T\Project\2. Data\1. Sound_samples\5. Full_recordings\All_Collected\Guitar_Models\LP', 
#    r'C:\Users\Kata\OneDrive - Aalborg Universitet\CE8-DESKTOP-T44IC5T\Project\2. Data\1. Sound_samples\5. Full_recordings\All_Collected\Guitar_Models\SC',
#    r'C:\Users\Kata\OneDrive - Aalborg Universitet\CE8-DESKTOP-T44IC5T\Project\2. Data\1. Sound_samples\5. Full_recordings\All_Collected\Guitar_Models\SG', 
#    r'C:\Users\Kata\OneDrive - Aalborg Universitet\CE8-DESKTOP-T44IC5T\Project\2. Data\1. Sound_samples\5. Full_recordings\All_Collected\Guitar_Models\TC'
#    ]

#audio = Dataset()
#audio.create_dataset(dirs)
#audio.scale()

#print(f'Training data : {audio.X_train}')
#print(f'Training label : {audio.y_train}')