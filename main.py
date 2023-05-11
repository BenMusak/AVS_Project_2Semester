import audio_lib as al
import KNN as knn
import numpy as np
import LDA as lda
import SVM as svm
#import remove_silence as rs
import utils
import os



def extract_features(audio_file, sr, label, res_val, diff_extremes_val, display=True):
    y = audio_file
    al.get_samples(y)
    MFCCs = al.extract_MFCCs(y, sr, res=13)
    Mel = al.extract_Mel(y, sr, res=128)

    if display:
        al.visualize_MFCCs_Mel(MFCCs, Mel, sr)

    train_x, train_y, test_x, test_y = al.dataset_split(MFCCs, Mel, sr, label, res=res_val, diff_extremes=diff_extremes_val, display=display)

    return train_x, train_y, test_x, test_y


def main():
    Multifile = True

    if Multifile:
        # Remove silence from an audio files
        #rs.remove_silence(r"C:\Users\Benja\Aalborg Universitet\AVS - Semester 8 - Group 841 - 2. Data\1. Sound_samples\5. Full_recordings\AudioStrumming_LP_Neck_Tone_4.wav")

        # Set the paths to the directories containing the sound files
        user = os.getlogin()
        directories = ["C:\\Users\\{0}\\Aalborg Universitet\\AVS - Semester 8 - Group 841 - Project\\2. Data\\1. Sound_samples\\5. Full_recordings\\All_data\\WAV\\Adjusted_for_clipping\\Les_Paul_(LP)".format(user), 
                       "C:\\Users\\{0}\\Aalborg Universitet\\AVS - Semester 8 - Group 841 - Project\\2. Data\\1. Sound_samples\\5. Full_recordings\\All_data\\WAV\\Adjusted_for_clipping\\Solid_Guitar_(SG)".format(user),
                       "C:\\Users\\{0}\\Aalborg Universitet\\AVS - Semester 8 - Group 841 - Project\\2. Data\\1. Sound_samples\\5. Full_recordings\\All_data\\WAV\\Adjusted_for_clipping\\Stratocaster_(SC)".format(user), 
                       "C:\\Users\\{0}\\Aalborg Universitet\\AVS - Semester 8 - Group 841 - Project\\2. Data\\1. Sound_samples\\5. Full_recordings\\All_data\\WAV\\Adjusted_for_clipping\\Telecaster_(TC)".format(user)]
                    #    r'C:\Users\jespe\Aalborg Universitet\AVS - Semester 8 - Group 841 - Project\2. Data\1. Sound_samples\6. Guitar_same_classes\LP_Bridge', 
                    #    r'C:\Users\jespe\Aalborg Universitet\AVS - Semester 8 - Group 841 - Project\2. Data\1. Sound_samples\6. Guitar_same_classes\LP_Neck'
        
        
        # Labels for the classes
        label_names = ["LP", "SG", "SC", "TC"]

        # Load Audio Files from the given directories
        strum_list = al.load_files_from_directories(directories)

        # Extract features
        train_x, train_y, test_x, test_y = [], [], [], []

        # Loop through the list of lists containing the audio files
        for label in range(len(strum_list)):
            print("lenght of strum_list: ", len(strum_list[label]))

            # Extract Mel and MFCC features from the current audio file
            mels, mffcs = al.extract_mel_mfcc_multible_files_no_sum(strum_list[label], label, display=False)

            # Split the data into train and test data
            train_x_, train_y_, test_x_, test_y_ = al.dataset_combine_multible_files(mffcs, mels, strum_list[label], label)

            # Append the data to the lists so that they can be concatenated later
            train_x.append(train_x_)
            train_y.append(train_y_)
            test_x.append(test_x_)
            test_y.append(test_y_)
        
        # Concatenate all the data
        train_x = np.concatenate(train_x)
        train_y = np.concatenate(train_y)
        test_x = np.concatenate(test_x)
        test_y = np.concatenate(test_y)

        print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)

        # Save Dataset
        al.save_dataset(train_x, train_y, test_x, test_y)

        # Perform LDA on the train data and reduce the dimensionality of the train and test data
        model_LDA, train_, train_tgt, test_, test_tgt = lda.LDA_Fishers(train_x, train_y, test_x, test_y, 3, label_names=label_names)
        model_KNN_LDA = knn.knn_model(train_, test_, train_tgt, test_tgt, "KNN with LDA")
        model_KNN = knn.knn_model(train_x, test_x, train_y, test_y, "KNN without LDA")
        model_SVM = svm.svm_model(train_x, test_x, train_y, test_y)

        # Save the model 
        #file_path_KNN = utils.save_model(model_KNN_LDA, "KNN_LDA_model")
        file_path_KNN = utils.save_model(model_KNN, "KNN_model")
        file_path_LDA = utils.save_model(model_LDA, "LDA_model")
        file_path_SVM = utils.save_model(model_SVM, "SVM_model")

        # Load the model
        #loaded_model = utils.load_model(file_path_SVM)
        #print("loaded model: ", loaded_model)
    else:
        # Load Audio Files
        SG_audio, SG_sr = al.load_audio_file("Test_dataset/AudioStrumming_SG.wav")
        SC_audio, SC_sr = al.load_audio_file("Test_dataset/AudioStrumming_SC.wav")
        SC_audio_Neck, SC_sr_Neck = al.load_audio_file("Test_dataset/AudioStrumming_SC.wav")
        SC_audio_fender, SC_sr_fender = al.load_audio_file("Test_dataset/AudioStrumming_SC_Fender.wav")
        LP_audio, LP_sr = al.load_audio_file("Test_dataset/AudioStrumming_LP.wav")

        # Extract Features
        SG_train_x, SG_train_y, SG_test_x, SG_test_y = extract_features(SG_audio, SG_sr, 1, 50, 43)
        SC_train_x, SC_train_y, SC_test_x, SC_test_y = extract_features(SC_audio, SC_sr, 2, 50, 59)
        SC_train_x_Neck, SC_train_y_Neck, SC_test_x_Neck, SC_test_y_Neck = extract_features(SC_audio_Neck, SC_sr_Neck, 2, 50, 59, display=True)
        SC_train_fender_x, SC_train_fender_y, SC_test_fender_x, SC_test_fender_y = extract_features(SC_audio_fender, SC_sr_fender, 2, 50, 93)
        LP_train_x, LP_train_y, LP_test_x, LP_test_y = extract_features(LP_audio, LP_sr, 3, 50, 62.1)

        # Combine datasets
        train_x = np.concatenate((SG_train_x, SC_train_x, LP_train_x, SC_train_x_Neck, SC_train_fender_x), axis=0)
        train_y = np.concatenate((SG_train_y, SC_train_y, LP_train_y, SC_train_y_Neck, SC_train_fender_y), axis=0)
        test_x = np.concatenate((SG_test_x, SC_test_x, LP_test_x, SC_test_x_Neck, SC_test_fender_x), axis=0)
        test_y = np.concatenate((SG_test_y, SC_test_y, LP_test_y, SC_test_y_Neck, SC_test_fender_y), axis=0)

        # Save Dataset
        al.save_dataset(train_x, train_y, test_x, test_y)

        # Run LDA to reduce dimensions and plot the data
        train_lda, test_lda = lda.LDA_Fishers(train_x, train_y, test_x, 2)

        # Run KNN model on dataset
        model = knn.knn_model(train_x, test_x, train_y, test_y)


if __name__ == "__main__":
    main()