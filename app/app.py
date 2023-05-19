import streamlit as st
import record as r
import LDA as lda
import utils
import os
from tqdm import tqdm
import remove_silence as rs
import pytorch_utils
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import plotly.express as px

# page configuration
page_icon = Image.open(r'app\mel_spec.jpg')
st.set_page_config(page_title="SuperCoolGuitar2000", page_icon=page_icon, layout="centered", initial_sidebar_state="auto", menu_items=None)

# containers
header = st.container() # title
record = st.container() # record audio and show audio wave

# Paths
LABELS = ["guitar_type", "pickup", "pickup_position", "strumming", "player"]
MODELS_PATH_KNN = ["models/KNN_model_{}.sav".format(label) for label in LABELS]
models_KNN = [utils.load_model(MODEL_PATH_KNN, LABELS) for MODEL_PATH_KNN in MODELS_PATH_KNN]
MODELS_PATH_SVM = ["models/SVM_model_{}.sav".format(label) for label in LABELS]
models_SVM = [utils.load_model(MODEL_PATH_SVM, LABELS) for MODEL_PATH_SVM in MODELS_PATH_SVM]
MODELS_PATH_LDA = ["models/LDA_model_{}.sav".format(label) for label in LABELS]
models_LDA = [utils.load_model(MODEL_PATH_LDA, LABELS) for MODEL_PATH_LDA in MODELS_PATH_LDA]
WAVE_PATH = r"record\output.wav"
DATASET_PATH = ["models/fishers_dataset/{}_LDA_Fishers_train.npz".format(label) for label in LABELS]
SCALAR_PATH = ["scaler_{}_KNN_without_LDA.pkl".format(label) for label in LABELS]
MODEL_PATH_CNN= r"models\CNN_model.ckpt"
model_CNN = utils.load_CNN_model(MODEL_PATH_CNN) #load KNN model
plot_LDA = False
#og_features, og_targets =  lda.load_dataset(DATASET_PATH)


with header:
    st.title('Guitar Classification 🎸')


with record: 
    rec_button = st.button('Record', key='audio-rec')
    predict_folder = st.button("Predict from folder", key="predict-folder")

    if rec_button:

        with st.spinner("Recording..."):
            st.balloons()

            # record audio
            r.record_audio(WAVE_PATH)

        with st.spinner("Removing Silience..."):
            # remove silence from record
            rs.remove_silence_from_single_file(WAVE_PATH)
            
            # import .wav audio file
            audio_data, sr = utils.load_file(WAVE_PATH, 44100)

            st.snow()

            # show audio wave if there is any data
            if audio_data is not None:
                st.line_chart(audio_data)
            else : 
                st.exception("No audio data file ☢️")
            
        with st.spinner("Predicting labels..."):
            # Convert audio data to features
            audio_data_mel, _ = utils.load_file(WAVE_PATH, 48000)
            mel_spectrogram = utils.convert_mel_spectrogram(audio_data_mel, 48000)

            # LDA, KNN, and SVM section
            st("KNN, SVM, and LDA")
            col_knn, col_svm, col_lda = st.columns(3)

            # Display the LDA model
            for i, model_LDA in enumerate(models_LDA):
                scalar_name = "models/scalars/scaler_{}_KNN_without_LDA.pkl".format(LABELS[i])
                __, unscaled_features = utils.convert_mfcc(audio_data, sr, scalar_name)
                features_LDA = model_LDA.transform(unscaled_features)
                y_pred_LDA = model_LDA.predict(unscaled_features)
                labels = utils.get_labels(LABELS[i])
                y_label_LDA = labels[y_pred_LDA[0]]
                col_lda.metric(label="LDA model for {}".format(LABELS[i]), percentage=y_label_LDA)
                og_features, og_targets =  lda.load_dataset(DATASET_PATH[i])

                if plot_LDA:
                    st.pyplot(lda.plot_data(og_features,  og_targets, "LDA model for {}".format(LABELS[i]), labels, features_LDA), clear_figure=True)
            
            # kNN prediction
            for i, model_KNN in enumerate(models_KNN):
                scalar_name = "models/scalars/scaler_{}_KNN_without_LDA.pkl".format(LABELS[i])
                features, unscaled_features = utils.convert_mfcc(audio_data, sr, scalar_name)
                y_pred_knn = model_KNN.predict(features)
                labels = utils.get_labels(LABELS[i])
                y_label_knn = labels[y_pred_knn[0]]
                col_knn.metric(label="KNN model for {}".format(LABELS[i]), percentage=y_label_knn)

            # SVM prediction
            for i, model_SVM in enumerate(models_SVM):
                scalar_name = "models/scalars/scaler_{}_KNN_without_LDA.pkl".format(LABELS[i])
                features, unscaled_features = utils.convert_mfcc(audio_data, sr, scalar_name)
                y_pred_svm = model_SVM.predict(features)
                labels = utils.get_labels(LABELS[i])
                y_label_svm = labels[y_pred_svm[0]]
                col_svm.metric("SVM model for {}".format(LABELS[i]), percentage=y_label_svm)

            # CNN classification
            st("CNN classifications")
            col_guitar_type, col_pickup = st.columns(2)
            col_pickup_position, col_strumming, col_player = st.columns(3)
            
            mel_spectrogram_t = pytorch_utils.convert_mel_spec_t(mel_spectrogram)
            y_pred_cnn = model_CNN(mel_spectrogram_t)
            predictions = pytorch_utils.multilabel_predictions(y_pred_cnn)
            
            col_guitar_type.metric(label="Guitar Type", percentage=pytorch_utils.get_guitar_type_labels()[predictions[1].item()])
            col_pickup.metric(label="Pickup", percentage=pytorch_utils.get_pickup_labels()[predictions[2].item()])
            col_pickup_position.metric(label="Pickup Position", percentage=pytorch_utils.get_pickup_position_labels()[predictions[3].item()])
            col_strumming.metric(label="Strumming", percentage=pytorch_utils.get_strumming_labels()[predictions[4].item()])
            col_player.metric(label="Player", percentage=pytorch_utils.get_player_labels()[predictions[5].item()])


    if predict_folder:

        st.title("Predict from folder")
        #folder_path = st.text_input("Folder path", key="folder-path")
        #st.write(folder_path)

        folder_path = r'app\test_samples_quick'

        with st.spinner("Predicting labels..."):
            audio_pred = []
            
            if folder_path:
                audio_files = os.listdir(folder_path)
                KNN_predictions = []
                SVM_predictions = []
                LDA_predictions = []

                for idx in tqdm(range(len(audio_files))):
                    audio_file = audio_files[idx]
                    audio_path = os.path.join(folder_path, audio_file)
                    audio_data, sr = utils.load_file(audio_path, 44100)

                    # Display the LDA model
                    for i, model_LDA in enumerate(models_LDA):
                        scalar_name = "models/scalars/scaler_{}_KNN_without_LDA.pkl".format(LABELS[i])
                        __, unscaled_features = utils.convert_mfcc(audio_data, sr, scalar_name)
                        features_LDA = model_LDA.transform(unscaled_features)
                        y_pred_LDA = model_LDA.predict(unscaled_features)
                        #labels = utils.get_labels(LABELS[i])
                        y_label_LDA = y_pred_LDA[0]
                        LDA_predictions.append(y_label_LDA)
                    
                    # kNN prediction
                    for i, model_KNN in enumerate(models_KNN):
                        scalar_name = "models/scalars/scaler_{}_KNN_without_LDA.pkl".format(LABELS[i])
                        features, unscaled_features = utils.convert_mfcc(audio_data, sr, scalar_name)
                        y_pred_knn = model_KNN.predict(features)
                        #labels = utils.get_labels(LABELS[i])
                        y_label_knn = y_pred_knn[0]
                        KNN_predictions.append(y_label_knn)
                        

                    # SVM prediction
                    for i, model_SVM in enumerate(models_SVM):
                        scalar_name = "models/scalars/scaler_{}_KNN_without_LDA.pkl".format(LABELS[i])
                        features, unscaled_features = utils.convert_mfcc(audio_data, sr, scalar_name)
                        y_pred_svm = model_SVM.predict(features)
                        #labels = utils.get_labels(LABELS[i])
                        y_label_svm = y_pred_svm[0]
                        SVM_predictions.append(y_label_svm)

                    # Add the predictions to the audio_pred list with the audio file name
                    audio_pred.append([audio_file, KNN_predictions, SVM_predictions, LDA_predictions])

                    # Reset the predictions
                    KNN_predictions = []
                    SVM_predictions = []
                    LDA_predictions = []
            
        # Calculate the accuracy of the predictions for each sound file
        with st.spinner("Calculating accuracy..."):
            
            # Load the CSV file with the correct labels
            df = pd.read_csv("app/metadata_jesper_testdata.csv", sep=";")

            for label_type in LABELS:
                label = utils.get_labels_reverse(label_type)
                df[label_type] = df[label_type].apply(lambda x : label.get(x))

            
            st.header("Scores")
            col_knn, col_svm, col_lda = st.columns(3)

            # Calculate the accuracy for each audio file
            KNN_total_acc = 0
            SVM_total_acc = 0
            LDA_total_acc = 0

            correct_labels_all = []
            predictions_KNN = []
            predictions_SVM = []
            predictions_LDA = []
            
            for audio in audio_pred:
                # Get the correct labels for the audio file
                correct_labels = df[df["name"] == audio[0]]
                correct_labels = correct_labels.drop(columns=["name"])
                correct_labels = correct_labels.values.astype(int).tolist()[0]
                correct_labels_all.append(correct_labels)
                
                # convert the predictions to a numpy array as int values for comparison with the correct labels
                KNN_report_guitar_type, KNN_report_pickup, KNN_report_pickup_pos, KNN_report_strumming, KNN_report_play = utils.get_predictions(audio[1])
                SVM_report_guitar_type, SVM_report_pickup, SVM_report_pickup_pos, SVM_report_strumming, SVM_report_play = utils.get_predictions(audio[2])
                LDA_report_guitar_type, LDA_report_pickup, LDA_report_pickup_pos, LDA_report_strumming, LDA_report_play = utils.get_predictions(audio[3])

                # Add the predictions to the list
                predictions_KNN.append([KNN_report_guitar_type, KNN_report_pickup, KNN_report_pickup_pos, KNN_report_strumming, KNN_report_play])
                predictions_SVM.append([SVM_report_guitar_type, SVM_report_pickup, SVM_report_pickup_pos, SVM_report_strumming, SVM_report_play])
                predictions_LDA.append([LDA_report_guitar_type, LDA_report_pickup, LDA_report_pickup_pos, LDA_report_strumming, LDA_report_play])


            # Get the report for each model
            KNN_report_guitar_type, KNN_report_pickup, KNN_report_pickup_pos, KNN_report_strumming, KNN_report_play = utils.get_reports(correct_labels_all, predictions_KNN)
            SVM_report_guitar_type, SVM_report_pickup, SVM_report_pickup_pos, SVM_report_strumming, SVM_report_play = utils.get_reports(correct_labels_all, predictions_SVM)
            LDA_report_guitar_type, LDA_report_pickup, LDA_report_pickup_pos, LDA_report_strumming, LDA_report_play = utils.get_reports(correct_labels_all, predictions_LDA)

            # Display the accuracy and f1-score for each model
            print('-------------')
            print(KNN_report_guitar_type)
            print('-------------')
            print(KNN_report_pickup)
            print('-------------')
            print(KNN_report_pickup_pos)
            print('-------------')
            print(KNN_report_strumming)
            print('-------------')
            print(KNN_report_play)

            guitar_type = ['LP', 'SG', 'SC', 'TC']
            pickup = ['Humbucker', 'Single Coil']
            pickup_pos = ['Bridge', 'Middle', 'Neck']
            strumming = ['Open', 'A-major']
            player = ['JM', 'VS', 'BH', 'JG', 'KB', 'AL']

            
            # score_type : precision, recall, f1-score
            SCORE_TYPE = 'precision'

            knn_guitar_type = utils.data_for_plot(KNN_report_guitar_type, guitar_type, SCORE_TYPE)
            knn_pickup = utils.data_for_plot(KNN_report_pickup, pickup, SCORE_TYPE)
            knn_pickup_pos = utils.data_for_plot(KNN_report_pickup_pos, pickup_pos, SCORE_TYPE)
            knn_strumming = utils.data_for_plot(KNN_report_strumming, strumming, SCORE_TYPE)
            knn_player = utils.data_for_plot(KNN_report_play, player, SCORE_TYPE)

            svm_guitar_type = utils.data_for_plot(SVM_report_guitar_type, guitar_type, SCORE_TYPE)
            svm_pickup = utils.data_for_plot(SVM_report_pickup, pickup, SCORE_TYPE)
            svm_pickup_pos = utils.data_for_plot(SVM_report_pickup_pos, pickup_pos, SCORE_TYPE)
            svm_strumming = utils.data_for_plot(SVM_report_strumming, strumming, SCORE_TYPE)
            svm_player = utils.data_for_plot(SVM_report_play, player, SCORE_TYPE)

            lda_guitar_type = utils.data_for_plot(LDA_report_guitar_type, guitar_type, SCORE_TYPE)
            lda_pickup = utils.data_for_plot(LDA_report_pickup, pickup, SCORE_TYPE)
            lda_pickup_pos = utils.data_for_plot(LDA_report_pickup_pos, pickup_pos, SCORE_TYPE)
            lda_strumming = utils.data_for_plot(LDA_report_strumming, strumming, SCORE_TYPE)
            lda_player = utils.data_for_plot(LDA_report_play, player, SCORE_TYPE)

            col_knn.subheader('KNN')
            col_knn.caption('Guitar type')
            col_knn.bar_chart(data=knn_guitar_type, x='label', y='percentage')
            col_knn.caption('Pickup')
            col_knn.bar_chart(data=knn_pickup, x='label', y='percentage')
            col_knn.caption('Pickup position')
            col_knn.bar_chart(data=knn_pickup_pos, x='label', y='percentage')
            col_knn.caption('Strumming')
            col_knn.bar_chart(data=knn_strumming, x='label', y='percentage')
            col_knn.caption('Player')
            col_knn.bar_chart(data=knn_player, x='label', y='percentage')
            
            col_svm.subheader('SVM')
            col_svm.caption('Guitar type')
            col_svm.bar_chart(data=svm_guitar_type, x='label', y='percentage')
            col_svm.caption('Pickup')
            col_svm.bar_chart(data=svm_pickup, x='label', y='percentage')
            col_svm.caption('Pickup position')
            col_svm.bar_chart(data=svm_pickup_pos, x='label', y='percentage')
            col_svm.caption('Strumming')
            col_svm.bar_chart(data=svm_strumming, x='label', y='percentage')
            col_svm.caption('Player')
            col_svm.bar_chart(data=svm_player, x='label', y='percentage')

            col_lda.subheader('LDA')
            col_lda.caption('Guitar type')
            col_lda.bar_chart(data=lda_guitar_type, x='label', y='percentage')
            col_lda.caption('Pickup')
            col_lda.bar_chart(data=lda_pickup, x='label', y='percentage')
            col_lda.caption('Pickup position')
            col_lda.bar_chart(data=lda_pickup_pos, x='label', y='percentage')
            col_lda.caption('Strumming')
            col_lda.bar_chart(data=lda_strumming, x='label', y='percentage')
            col_lda.caption('Player')
            col_lda.bar_chart(data=lda_player, x='label', y='percentage')

