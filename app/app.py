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

# variables
MODEL_PATH_KNN = r"models\KNN_model.sav"
MODEL_PATH_SVM = r"models\SVM_model.sav"
MODEL_PATH_LDA = r"models\LDA_model.sav"
WAVE_PATH = r"record\output.wav"
DATASET_PATH = r"LDA_Fishers_train.npz"
SCALAR_PATH = r"scaler.pkl"
MODEL_PATH_CNN= r"models\CNN_model.ckpt"
model_KNN, model_LDA, model_SVM, model_CNN = utils.load_model(MODEL_PATH_KNN, MODEL_PATH_LDA, MODEL_PATH_SVM, MODEL_PATH_CNN) #load KNN model
og_features, og_targets = lda.load_dataset(DATASET_PATH)

# label for dataset
label_guitar_model = {
    0 : 'Les Paul',
    1 : 'Solid Guitar', 
    2 : 'Stratocaster',
    3 : 'TeleCaster'
    }

#################################
##          DE APP             ##
#################################

with header:
    st.title('Guitar Classification 🎸')

with record: 
    rec_button = st.button('Record', key='audio-rec')
    
    if rec_button:

        with st.spinner("Loading..."):
            st.balloons()

            # record audio
            #r.record_audio(WAVE_PATH)

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
            
            # Convert audio data to features
            features, unscaled_features, fig_MFCCs = utils.convert_mfcc(audio_data, sr, SCALAR_PATH)
            mel_spectrogram = utils.convert_mel_spectrogram(audio_data, sr)

            # display Mel and MFCCs
            st.subheader("Mel-Spectogram and MFCCs")
            st.pyplot(fig_MFCCs, clear_figure=True)
            
            # Display the LDA model
            features_LDA = model_LDA.transform(features)
            print("LDA feature shape: ", features_LDA.shape)
            st.subheader('LDA model and prediction')
            y_pred_LDA = model_LDA.predict(features)
            y_label_LDA = label_guitar_model[y_pred_LDA[0]]
            col_LDA, _ = st.columns(2)
            col_LDA.metric(label="LDA", value=y_label_LDA)
            st.pyplot(lda.plot_data(og_features, og_targets, "LDA model", label_guitar_model, features_LDA), clear_figure=True)

            # kNN and SVM section
            st.subheader("KNN and SVM")
            col_knn, col_svm= st.columns(2)
            
            # kNN prediction
            y_pred_knn = model_KNN.predict(features)
            y_label_knn = label_guitar_model[y_pred_knn[0]]
            col_knn.metric(label="KNN", value=y_label_knn)

            # SVM prediction
            y_pred_svm = model_SVM.predict(features)
            y_label_svm = label_guitar_model[y_pred_svm[0]]
            col_svm.metric(label="SVM", value=y_label_svm)

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
