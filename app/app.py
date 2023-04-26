import streamlit as st
import dataset as ds
import record as r
import utils
import remove_silence as rs

# containers
header = st.container() # title
record = st.container() # record audio and show audio wave
clf = st.container() # results of classifications

CLASSIFY = False # doing classification
MODEL_PATH = r"AVS_Project_2Semester\models\KNN_model.sav"
model = utils.load_model(MODEL_PATH) #load KNN model
WAVE_PATH = r"C:\Users\Benja\Documents\Skole\AI-Vision-Sound\8th_Semester\Project\code\AVS_Project_2Semester\record\output.wav"

# label for dataset
label_guitar_model = {
    0 : 'Les Paul',
    1 : 'StratoCaster', 
    2 : 'Solid Guitar',
    3 : 'TeleCaster'
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
    features = utils.convert(audio_data, sr)
    
    if clf_button:
        st.write("Data converted")
        
        # make kNN prediction
        y_pred = model.predict(features)
        y_label = label_guitar_model[y_pred[0]]
        st.metric(label="Class", value=y_label)
        
        #CLASSIFY = False
        clf_button = False