from pydub import AudioSegment
from pydub.silence import split_on_silence

def remove_silence_from_single_file(audio_path):
    
    # Load the audio file
    sound_file = AudioSegment.from_wav(audio_path)

    # Split the audio file on non-silent parts
    audio_parts = split_on_silence(sound_file, min_silence_len=150, silence_thresh=-42)

    output_folder = audio_path

    for i, audio_part in enumerate(audio_parts):
        audio_part.export(output_folder, format='wav')