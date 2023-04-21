import audio_lib as al
import dataset as ds
import model as m

# labels
label_guitar_model = {
    0 : 'LP',
    1 : 'SC', 
    2 : 'SG',
    3 : 'TC'
    }

# create dataset
dirs = [
    r'C:\Users\Kata\OneDrive - Aalborg Universitet\CE8-DESKTOP-T44IC5T\Project\2. Data\1. Sound_samples\5. Full_recordings\All_Collected\Guitar_Models\LP', 
    r'C:\Users\Kata\OneDrive - Aalborg Universitet\CE8-DESKTOP-T44IC5T\Project\2. Data\1. Sound_samples\5. Full_recordings\All_Collected\Guitar_Models\SC',
    r'C:\Users\Kata\OneDrive - Aalborg Universitet\CE8-DESKTOP-T44IC5T\Project\2. Data\1. Sound_samples\5. Full_recordings\All_Collected\Guitar_Models\SG', 
    r'C:\Users\Kata\OneDrive - Aalborg Universitet\CE8-DESKTOP-T44IC5T\Project\2. Data\1. Sound_samples\5. Full_recordings\All_Collected\Guitar_Models\TC'
    ]

audio = ds.Dataset()
audio.create_dataset(dirs)
audio.scale() # scale dataset

# model and training
model = m.KNN()
model.train(audio.X_train, audio.y_train)

# prediction
print(audio.X_test[0])
y_pred = model.predict([audio.X_test[0]])

y_true = audio.y_test[0]
print(f'Predicted label : {y_pred[0]} - {label_guitar_model[y_pred[0]]}' )
print(f'True label : {y_true} - {label_guitar_model[y_true]}')
