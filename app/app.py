import streamlit as st
import record as r

header = st.container() # just title
record = st.container() # record audio and show audio wave


with header:
    header.title('Guitar Classification 🎸')

with record:
    record.title('Record audio') # TODO : remove later
    
    rec_button = record.button('Record', key='audio-rec')
    record.write(rec_button)
    #record.line_chart(data)
    
    if rec_button:
        data = r.record_audio()
        record.line_chart(data)
        rec_button = False

    
    
    # TODO : visualize audio wave

    # if button pressed
    # record audio -> pyaudio
    
