import os
import librosa


def load_files_from_directories():
    # Set the paths to the directories containing the sound files
    directories = [r'C:\Users\jespe\Desktop\Uni\8. Semester\Project\1. SG_sound_samples', r'C:\Users\jespe\Desktop\Uni\8. Semester\Project\2. SC_sound_samples', r'C:\Users\jespe\Desktop\Uni\8. Semester\Project\3. LP_sound_samples', r'C:\Users\jespe\Desktop\Uni\8. Semester\Project\4. TC_sound_samples']

    # Loop over the directories and get a list of all the sound files in each directory
    sound_files = []
    for directory in directories:
        sound_files += [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.wav')]

    # Loop over the sound files and load them using librosa.load
    for sound_file in sound_files:
        # Load the sound file using librosa.load
        y, sr = librosa.load(sound_file, sr=44100, duration=2)

    return y, sr  
