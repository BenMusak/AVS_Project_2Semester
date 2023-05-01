import streamlit as st
import dataset as ds
import record as r
import LDA as lda
import utils
import remove_silence as rs

# containers
header = st.container() # title
record = st.container() # record audio and show audio wave
clf = st.container() # results of classifications

CLASSIFY = False # doing classification
MODEL_PATH_KNN = r"AVS_Project_2Semester\models\KNN_model.sav"
MODEL_PATH_LDA = r"AVS_Project_2Semester\models\LDA_model.sav"
WAVE_PATH = r"C:\Users\Benja\Documents\Skole\AI-Vision-Sound\8th_Semester\Project\code\AVS_Project_2Semester\record\output.wav"
DATASET_PATH = r"LDA_Fishers_train.npz"
model_KNN, model_LDA = utils.load_model(MODEL_PATH_KNN, MODEL_PATH_LDA) #load KNN model
og_features, og_targets = lda.load_dataset(DATASET_PATH)

# label for dataset
label_guitar_model = {
    0 : 'Les Paul',
    1 : 'Solid Guitar', 
    2 : 'Stratocaster',
    3 : 'TeleCaster',
    4 : 'TeleCaster_Hum',
    5 : 'Unkown'
    }

#################################

with header:
    st.title('Guitar Classification 🎸')

with record:
    st.header('Record audio') # TODO : remove later maybe
    
    rec_button = st.button('Record', key='audio-rec', on_click=r.record_audio())
    st.write(rec_button) # TODO : comment out in the end

    rs.remove_silence_from_single_file(WAVE_PATH)

    audio_data, sr = utils.load_file(WAVE_PATH)

    # show audio wave if record button was pressed
    if rec_button:
        st.line_chart(audio_data)


with clf: 
    st.header('Classification') # TODO : remove later maybe

    clf_button = st.button('Classify', key='knn-clf') # TODO : remove later
    
    if clf_button:
        st.write("Converting Data...")
        audio_data, sr = utils.load_file(WAVE_PATH)

        # make kNN prediction
        features = utils.convert(audio_data, sr)
        y_pred = model_KNN.predict(features)
        print(y_pred)
        print(y_pred[0])
        y_label = label_guitar_model[y_pred[0]]
        st.metric(label="Class", value=y_label)

        # Display the LDA model
        features_LDA = model_LDA.transform(features)
        st.subheader('LDA model')
        st.pyplot(lda.plot_data(og_features, og_targets, "LDA model", label_guitar_model, features_LDA))
        
        #CLASSIFY = False
        clf_button = False