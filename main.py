import audio_lib as al
import KNN as knn
import numpy as np
import LDA as lda
import remove_silence as rs



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
    jesper = False

    # Jesper sound extractor
    if jesper:
        input_folder = r"C:\Users\Benja\Aalborg Universitet\AVS - Semester 8 - Group 841 - 2. Data\1. Sound_samples\5. Full_recordings"
        rs.load_sound_files(input_folder)
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