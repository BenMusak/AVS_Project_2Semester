import matplotlib.pyplot as plt
import numpy as np
import wave
from pydub import AudioSegment
from pydub.silence import split_on_silence
import os



def display_waveform(Audio_file):
    # Open the audio file in read-only mode
    with wave.open(Audio_file, 'rb') as audio_file:

        # Get the audio file parameters
        num_channels = audio_file.getnchannels()
        print("Number of channels: ", num_channels)
        sample_rate = audio_file.getframerate()
        print("sample rate: ", sample_rate)
        num_frames = audio_file.getnframes()
        print("num frames: ", num_frames)
        sample_width = audio_file.getsampwidth()

        # Read the audio frames as bytes
        audio_frames = audio_file.readframes(num_frames)

        # Convert the audio frames to a numpy array
        audio_signal = np.frombuffer(audio_frames, dtype=np.int16)

        # If the audio has multiple channels, average them
        if num_channels > 1:
            audio_signal = np.mean(audio_signal.reshape(-1, num_channels), axis=1)

        # Calculate the duration of the audio file in seconds
        duration = num_frames / float(sample_rate)

        # Create a time array based on the audio sample rate
        time_array = np.arange(0, duration, 1/sample_rate)

        # Plot the waveform
        plt.plot(time_array, audio_signal)
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title('Waveform of Audio Signal')
        plt.show()


# read from 4. full recordings
# save to each respective folder based on name


def remove_silence(audio_name):
    print("input: ", audio_name)


    # Load the audio file
    sound_file = AudioSegment.from_wav(audio_name)


    # Split the audio file on non-silent parts
    audio_parts = split_on_silence(sound_file, min_silence_len=150, silence_thresh=-42)

    # Loop through the audio parts and save them as separate files
    for i, audio_part in enumerate(audio_parts):

        filename = os.path.splitext(os.path.basename(audio_name))[0] + '_' + str(i+1) + '.wav'
        print("file_name1: ", filename)


        if "SG" in filename:
            # Export the audio part as a new file
            output_folder = os.path.join(r"C:\Users\jespe\Desktop\Uni\8. Semester\Project\1. SG_sound_samples", filename)
            audio_part.export(output_folder, format='wav')
        elif "SC" in filename:
            # Export the audio part as a new file
            output_folder = os.path.join(r"C:\Users\jespe\Desktop\Uni\8. Semester\Project\2. SC_sound_samples", filename)
            audio_part.export(output_folder, format='wav')
        elif "LP" in filename:
            # Export the audio part as a new file
            output_folder = os.path.join(r"C:\Users\jespe\Desktop\Uni\8. Semester\Project\3. LP_sound_samples", filename)
            audio_part.export(output_folder, format='wav')
        elif "TC" in filename:
            # Export the audio part as a new file
            output_folder = os.path.join(r"C:\Users\jespe\Desktop\Uni\8. Semester\Project\4. TC_sound_samples", filename)
            audio_part.export(output_folder, format='wav')

def load_sound_files(input_folder):
    # Loop through the files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".wav"):
            # Construct the full path of the input file
            input_path = os.path.join(input_folder, filename)
            print("input_path: ", input_path)
            remove_silence(input_path)

    




if __name__ == "__main__":
    input_folder = r"C:\Users\jespe\Desktop\Uni\8. Semester\Project\5. Full_recordings"

    load_sound_files(input_folder)



    # display_waveform(Audio_file)