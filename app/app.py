import streamlit as st
import dataset as ds
import model as m
import record as r

header = st.container() # just title
record = st.container() # record audio and show audio wave
knn_clf = st.container() # results of classification

CLASSIFY = False # doing classification

#################################
# TODO : if possible make this part somehow visible, e.g. loading bar

# load dataset
label_guitar_model = {
    'LP' : 0,
    'SC' : 1, 
    'SG' : 2,
    'TC' : 3
    }

dirs = [
    r'C:\Users\Kata\OneDrive - Aalborg Universitet\CE8-DESKTOP-T44IC5T\Project\2. Data\1. Sound_samples\5. Full_recordings\All_Collected\Guitar_Models\LP', 
    r'C:\Users\Kata\OneDrive - Aalborg Universitet\CE8-DESKTOP-T44IC5T\Project\2. Data\1. Sound_samples\5. Full_recordings\All_Collected\Guitar_Models\SC',
    r'C:\Users\Kata\OneDrive - Aalborg Universitet\CE8-DESKTOP-T44IC5T\Project\2. Data\1. Sound_samples\5. Full_recordings\All_Collected\Guitar_Models\SG', 
    r'C:\Users\Kata\OneDrive - Aalborg Universitet\CE8-DESKTOP-T44IC5T\Project\2. Data\1. Sound_samples\5. Full_recordings\All_Collected\Guitar_Models\TC'
    ]

audio= ds.Dataset()
audio.create_dataset(dirs)
audio.scale() # scale dataset


# model and training
model = m.KNN()
model.train(audio.X_train, audio.y_train)

#################################

with header:
    header.title('Guitar Classification 🎸')

with record:
    record.title('Record audio') # TODO : remove later
    
    rec_button = record.button('Record', key='audio-rec')
    record.write(rec_button) # TODO : comment out in the end
    #record.line_chart(data)
    
    if rec_button:
        data = r.record_audio()
        record.line_chart(data)
        rec_button = False
        CLASSIFY = True


# TODO : convert record to classification

with knn_clf: 
    knn_clf.title('kNN Classification') # TODO : remove later

    if CLASSIFY:
        CLASSIFY = False
        # make kNN prediction

        y_pred = model.predict(data)
        pass