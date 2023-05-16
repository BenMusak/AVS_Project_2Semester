import streamlit as st
import record as r
import LDA as lda
import utils
import os
import remove_silence as rs
import pytorch_utils
import numpy as np
from PIL import Image
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


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
#og_features, og_targets =  lda.load_dataset(DATASET_PATH)


with header:
    st.title('Guitar Classification 🎸')

with record: 
    rec_button = st.button('Record', key='audio-rec')
    
    if rec_button:

        with st.spinner("Recording..."):
            st.balloons()

            # record audio
            r.record_audio(WAVE_PATH)


        with st.spinner("Removing Silience..."):
            # remove silence from record
            rs.remove_silence_from_single_file(WAVE_PATH)
            
            # import .wav audio file
            audio_data, sr = utils.load_file(WAVE_PATH)

            st.snow()

            # show audio wave if there is any data
            if audio_data is not None:
                st.line_chart(audio_data)
            else : 
                st.exception("No audio data file ☢️")
            
        with st.spinner("Predicting labels..."):
            # Convert audio data to features
            #features, unscaled_features, fig_MFCCs = utils.convert_mfcc(audio_data, sr, SCALAR_PATH)
            mel_spectrogram = utils.convert_mel_spectrogram(audio_data, sr)

            # display Mel and MFCCs
            #st.subheader("Mel-Spectogram and MFCCs")
            #st.pyplot(fig_MFCCs, clear_figure=True)


            # LDA, KNN, and SVM section
            st.subheader("KNN, SVM, and LDA")
            col_knn, col_svm, col_lda = st.columns(3)

            # Display the LDA model
            for i, model_LDA in enumerate(models_LDA):
                scalar_name = "models/scalars/scaler_{}_KNN_without_LDA.pkl".format(LABELS[i])
                __, unscaled_features, fig_MFCCs = utils.convert_mfcc(audio_data, sr, scalar_name)
                #st.subheader("LDA model for {}".format(LABELS[i]))
                features_LDA = model_LDA.transform(unscaled_features)
                y_pred_LDA = model_LDA.predict(unscaled_features)
                labels = utils.get_labels(LABELS[i])
                y_label_LDA = labels[y_pred_LDA[0]]
                col_lda.metric(label="LDA model for {}".format(LABELS[i]), value=y_label_LDA)
                og_features, og_targets =  lda.load_dataset(DATASET_PATH[i])

                st.pyplot(lda.plot_data(og_features,  og_targets, "LDA model for {}".format(LABELS[i]), labels, features_LDA), clear_figure=True)

                #try:
                    #st.pyplot(lda.plot_data(og_features,  og_targets, "LDA model for {}".format(LABELS[i]), labels, features_LDA), clear_figure=True)
                #except ValueError as e:
                #    st.exception(e)
            
            # kNN prediction
            for i, model_KNN in enumerate(models_KNN):
                scalar_name = "models/scalars/scaler_{}_KNN_without_LDA.pkl".format(LABELS[i])
                features, unscaled_features, fig_MFCCs = utils.convert_mfcc(audio_data, sr, scalar_name)
                y_pred_knn = model_KNN.predict(features)
                labels = utils.get_labels(LABELS[i])
                y_label_knn = labels[y_pred_knn[0]]
                col_knn.metric(label="KNN model for {}".format(LABELS[i]), value=y_label_knn)

            # SVM prediction
            for i, model_SVM in enumerate(models_SVM):
                scalar_name = "models/scalars/scaler_{}_KNN_without_LDA.pkl".format(LABELS[i])
                features, unscaled_features, fig_MFCCs = utils.convert_mfcc(audio_data, sr, scalar_name)
                y_pred_svm = model_SVM.predict(features)
                labels = utils.get_labels(LABELS[i])
                y_label_svm = labels[y_pred_svm[0]]
                col_svm.metric("SVM model for {}".format(LABELS[i]), value=y_label_svm)

            # CNN classification
            st.subheader("CNN classifications")
            col_manu, col_guitar_type, col_pickup = st.columns(3)
            col_pickup_position, col_strumming, col_player = st.columns(3)
            
            mel_spectrogram_t = pytorch_utils.convert_mel_spec_t(mel_spectrogram)
            y_pred_cnn = model_CNN(mel_spectrogram_t)
            predictions = pytorch_utils.multilabel_predictions(y_pred_cnn)
            
            col_manu.metric(label="Manufacturer", value=pytorch_utils.get_manufacturer_labels()[predictions[0].item()])
            col_guitar_type.metric(label="Guitar Type", value=pytorch_utils.get_guitar_type_labels()[predictions[1].item()])
            col_pickup.metric(label="Pickup", value=pytorch_utils.get_pickup_labels()[predictions[2].item()])
            col_pickup_position.metric(label="Pickup Position", value=pytorch_utils.get_pickup_position_labels()[predictions[3].item()])
            col_strumming.metric(label="Strumming", value=pytorch_utils.get_strumming_labels()[predictions[4].item()])
            col_player.metric(label="Player", value=pytorch_utils.get_player_labels()[predictions[5].item()])
