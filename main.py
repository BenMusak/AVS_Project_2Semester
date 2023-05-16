import utils
import os
import vini_utils
import audio_lib as al
import KNN as knn
import numpy as np
import LDA as lda
import SVM as svm
from sklearn.model_selection import train_test_split



def extract_features(audio_file, sr, label, res_val, diff_extremes_val, display=True):
    y = audio_file
    al.get_samples(y)
    MFCCs = al.extract_MFCCs(y, sr, res=13)
    Mel = al.extract_Mel(y, sr, res=128)

    if display:
        al.visualize_MFCCs_Mel(MFCCs, Mel, sr)

    train_x, train_y, test_x, test_y = al.dataset_split(MFCCs, Mel, sr, label, res=res_val, diff_extremes=diff_extremes_val, display=display)

    return train_x, train_y, test_x, test_y


user = os.getlogin()
DATA_DIR = "C:\\Users\\{0}\\Aalborg Universitet\\AVS - Semester 8 - Group 841 - Project\\2. Data\\3. All_Sliced_Samples_with_CSV (No Clipping and Short samples removed)\\data".format(user) #Directory containing WAV files
CSV_DIR = "C:\\Users\\{0}\\Aalborg Universitet\\AVS - Semester 8 - Group 841 - Project\\2. Data\\3. All_Sliced_Samples_with_CSV (No Clipping and Short samples removed)\\metadata.csv".format(user)
SR = 44100
GLO_LABELS = ['player']


def main():
    Multifile = True

    if Multifile:

        data, labels, names = vini_utils.load_data_from_csv(DATA_DIR, CSV_DIR)

        for LABEL in GLO_LABELS:
            
            print("LABEL: ", LABEL)

            # Labels is a list of dictionaries. The dictionary shows the classes to the WAV file with the same index.
            # We cannot pass a list of dictionaries to the model, so we convert the dictionary to the desired label
            labels_des = vini_utils.list_get_desired_label(labels, LABEL)

            #MFCCs
            mfccs = vini_utils.extract_mfccs(data, sr=SR)

            #Split to train and test
            train_x, test_x = train_test_split(mfccs, test_size=0.2, random_state = 69)
            train_y, test_y = train_test_split(labels_des, test_size=0.2, random_state = 69)
            #train_names, test_names = train_test_split(names, test_size=0.2, random_state = 69)

            # Save Dataset
            al.save_dataset(train_x, train_y, test_x, test_y)

            # Get the label names
            if LABEL == 'manufacturer':
                label_names = vini_utils.get_manufacturer_labels()
            elif LABEL == 'guitar_type':
                label_names = vini_utils.get_guitar_type_labels()
            elif LABEL == 'pickup':
                label_names = vini_utils.get_pickup_labels()
            elif LABEL == 'pickup_position':
                label_names = vini_utils.get_pickup_position_labels()
            elif LABEL == 'strumming':
                label_names = vini_utils.get_strumming_labels()
            elif LABEL == 'player':
                label_names = vini_utils.get_player_labels()

            # Make sure the dimensions for the LDA reduction is the same as the number of classes - 1
            if len(label_names) <= 3:
                dim = len(label_names) - 1
            else:
                dim = 3

            print("LDA dimensionality for {}: {}".format(LABEL, dim))

            # Perform LDA on the train data and reduce the dimensionality of the train and test data
            model_LDA, train_, train_tgt, test_, test_tgt = lda.LDA_Fishers(train_x, train_y, test_x, test_y, dim, LABEL, label_names=label_names)
            file_path_LDA = utils.save_model(model_LDA, "LDA_model_{}".format(LABEL))

            model_KNN_LDA = knn.knn_model(train_, test_, train_tgt, test_tgt, label_names, "{}_KNN_with_LDA".format(LABEL))
            file_path_KNN = utils.save_model(model_KNN_LDA, "{}KNN_with_LDA".format(LABEL))

            model_KNN = knn.knn_model(train_x, test_x, train_y, test_y, label_names, "{}_KNN_without_LDA".format(LABEL))
            file_path_KNN = utils.save_model(model_KNN, "KNN_model_{}".format(LABEL))

            model_SVM_LDA = svm.svm_model(train_x, test_x, train_y, test_y, label_names, "{}_SVM_with_LDA".format(LABEL))
            file_path_SVM = utils.save_model(model_SVM_LDA, "{}SVM_with_LDA".format(LABEL))

            model_SVM = svm.svm_model(train_x, test_x, train_y, test_y, label_names, "{}_SVM_without_LDA".format(LABEL))
            file_path_SVM = utils.save_model(model_SVM, "SVM_model_{}".format(LABEL))
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