from pydub import AudioSegment
from pydub.silence import split_on_silence
import os



def remove_silence(audio_name):
    print("input: ", audio_name)

    # Load the audio file
    sound_file = AudioSegment.from_wav(audio_name)

    # Split the audio file on non-silent parts
    audio_parts = split_on_silence(sound_file, min_silence_len=150, silence_thresh=-42)

    # Loop through the audio parts and save them as separate files
    for i, audio_part in enumerate(audio_parts):

        filename = os.path.splitext(os.path.basename(audio_name))[0] + '_' + str(i+1) + '.wav'
        print("file_name: ", filename)


        if "SG" in filename:
            # Export the audio part as a new file
            output_folder = os.path.join(r"C:\Users\Benja\Aalborg Universitet\AVS - Semester 8 - Group 841 - 2. Data\1. Sound_samples\1. SG_sound_samples", filename)
            audio_part.export(output_folder, format='wav')
        elif "SC" in filename:
            # Export the audio part as a new file
            output_folder = os.path.join(r"C:\Users\Benja\Aalborg Universitet\AVS - Semester 8 - Group 841 - 2. Data\1. Sound_samples\2. SC_sound_samples", filename)
            audio_part.export(output_folder, format='wav')
        elif "LP" in filename:
            # Export the audio part as a new file
            output_folder = os.path.join(r"C:\Users\Benja\Aalborg Universitet\AVS - Semester 8 - Group 841 - 2. Data\1. Sound_samples\3. LP_sound_samples", filename)
            audio_part.export(output_folder, format='wav')
        elif "TC" in filename:
            # Export the audio part as a new file
            output_folder = os.path.join(r"C:\Users\Benja\Aalborg Universitet\AVS - Semester 8 - Group 841 - 2. Data\1. Sound_samples\4. TC_sound_samples", filename)
            audio_part.export(output_folder, format='wav')


def load_sound_files(input_folder):
    # Loop through the files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".wav"):
            # Construct the full path of the input file
            input_path = os.path.join(input_folder, filename)
            print("input_path: ", input_path)
            remove_silence(input_path)


def remove_silence_new_data(audio_name):
    print("input: ", audio_name)

    # Load the audio file
    sound_file = AudioSegment.from_wav(audio_name)

    # Split the audio file on non-silent parts
    audio_parts = split_on_silence(sound_file, min_silence_len=150, silence_thresh=-42)

    # Loop through the audio parts and save them as separate files
    for i, audio_part in enumerate(audio_parts):

        filename = os.path.splitext(os.path.basename(audio_name))[0] + '_' + str(i+1) + '.wav'
        print("file_name: ", filename)

        if "SC" in filename:
            # Export the audio part as a new file
            output_folder = os.path.join(r"C:\Users\Benja\Aalborg Universitet\AVS - Semester 8 - Group 841 - 2. Data\1. Sound_samples\5. Full_recordings\Kata_MusicTribe\WAV\Strums\SC", filename)
            audio_part.export(output_folder, format='wav')
        elif "TC" in filename:
            # Export the audio part as a new file
            output_folder = os.path.join(r"C:\Users\Benja\Aalborg Universitet\AVS - Semester 8 - Group 841 - 2. Data\1. Sound_samples\5. Full_recordings\Kata_MusicTribe\WAV\Strums\TC", filename)
            audio_part.export(output_folder, format='wav')
        elif "SG" in filename:
            # Export the audio part as a new file
            output_folder = os.path.join(r"C:\Users\Benja\Aalborg Universitet\AVS - Semester 8 - Group 841 - 2. Data\1. Sound_samples\5. Full_recordings\Kata_MusicTribe\WAV\Strums\SG", filename)
            audio_part.export(output_folder, format='wav')