import audio_lib as al
import KNN as knn
import numpy as np
import LDA as lda
import remove_silence as rs
import dataset as ds


def extract_features(audio_file, sr, label, res_val, diff_extremes_val, display=False):
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
        #rs.remove_silence_new_data(r"C:\Users\Benja\Aalborg Universitet\AVS - Semester 8 - Group 841 - 2. Data\1. Sound_samples\5. Full_recordings\Kata_MusicTribe\WAV\Individual\AudioStrumming_Supreme_SG_Neck_Open_KB.WAV")

        # Load Audio Files
        train_x, train_y, test_x, test_y = ds.create_dataset(r"C:\Users\Benja\Documents\Skole\AI-Vision-Sound\Competition\data\training.npy", r"C:\Users\Benja\Documents\Skole\AI-Vision-Sound\Competition\data\training_labels.npy")

        knn.knn_model(train_x, test_x, train_y, test_y)

        exit()
 
        directories = [r'C:\Users\Benja\Aalborg Universitet\AVS - Semester 8 - Group 841 - 2. Data\1. Sound_samples\5. Full_recordings\All_Collected\Strummer_Name\Jacobo',
                       r'C:\Users\Benja\Aalborg Universitet\AVS - Semester 8 - Group 841 - 2. Data\1. Sound_samples\5. Full_recordings\All_Collected\Strummer_Name\Kata',
                       r'C:\Users\Benja\Aalborg Universitet\AVS - Semester 8 - Group 841 - 2. Data\1. Sound_samples\5. Full_recordings\All_Collected\Strummer_Name\Vini']
        
        label_names = ["Jacobo", "Kata", "Vini"]

        # Load Audio Files
        strum_list = al.load_files_from_directories(directories)

        # Extract features
        train_x, train_y, test_x, test_y = [], [], [], []
        for label in range(len(strum_list)):
            print("lenght of strum_list: ", len(strum_list[label]))
            mels, mffcs = al.extract_mel_mfcc_multible_files(strum_list[label], label, display=False)
            train_x_, train_y_, test_x_, test_y_ = al.dataset_combine_multible_files(mffcs, mels, strum_list[label], label)
            train_x.append(train_x_)
            train_y.append(train_y_)
            test_x.append(test_x_)
            test_y.append(test_y_)
        
        # Concatenate all the data
        train_x = np.concatenate(train_x)
        train_y = np.concatenate(train_y)
        test_x = np.concatenate(test_x)
        test_y = np.concatenate(test_y)

        lda.LDA_Fishers(train_x, train_y, test_x, 2, label_names=label_names)

        knn.knn_model(train_x, test_x, train_y, test_y)

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
        knn.knn_model(train_x, test_x, train_y, test_y)


if __name__ == "__main__":
    main()