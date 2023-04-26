import streamlit as st
import dataset as ds
# import model as m
import record as r
import convert as c
#import utils 

header = st.container() # just title
record = st.container() # record audio and show audio wave
clf = st.container() # results of classification

CLASSIFY = False # doing classification

#################################
# load dataset
label_guitar_model = {
    0 : 'LP',
    1 : 'SC', 
    2 : 'SG',
    3 : 'TC'
    }

#dirs = [
#    r'C:\Users\Kata\OneDrive - Aalborg Universitet\CE8-DESKTOP-T44IC5T\Project\2. Data\1. Sound_samples\5. Full_recordings\All_Collected\Guitar_Models\LP', 
#    r'C:\Users\Kata\OneDrive - Aalborg Universitet\CE8-DESKTOP-T44IC5T\Project\2. Data\1. Sound_samples\5. Full_recordings\All_Collected\Guitar_Models\SC',
#    r'C:\Users\Kata\OneDrive - Aalborg Universitet\CE8-DESKTOP-T44IC5T\Project\2. Data\1. Sound_samples\5. Full_recordings\All_Collected\Guitar_Models\SG', 
#    r'C:\Users\Kata\OneDrive - Aalborg Universitet\CE8-DESKTOP-T44IC5T\Project\2. Data\1. Sound_samples\5. Full_recordings\All_Collected\Guitar_Models\TC'
#    ]

#audio= ds.Dataset()
#audio.create_dataset(dirs)
#audio.scale() # scale dataset


# model and training
#model = m.KNN()
#model.train(audio.X_train, audio.y_train)
#FILE_PATH = ""
#model = utils.load_model(FILE_PATH)

#################################

with header:
    st.title('Guitar Classification 🎸')

with record:
    st.title('Record audio') # TODO : remove later
    
    rec_button = st.button('Record', key='audio-rec', on_click=r.record_audio())
    st.write(rec_button) # TODO : comment out in the end
    
    audio_data, sr = c.load_file()
    
    # show linechart if record button was pressed
    if rec_button:
        st.line_chart(audio_data)
    

# TODO : convert record to classification
with clf: 
    st.title('Classification') # TODO : remove later

    clf_button = st.button('Convert', key='knn-clf') # TODO : remove later
    mfcc_data = c.convert(audio_data, sr)
    
    if clf_button:
        #mfcc_data = c.convert(audio_data, sr)
        st.write("Data converted")
        clf_button = False

        # make kNN prediction
        #y_pred = model.predict(data)
        #y_label = label_guitar_model[y_pred]
#        st.metric(label="Class", value=y_label)
#        CLASSIFY = False
#        pass