# add_gaps.py
# CS 6140: Final Project - Audio Inpainting
# Written on 03/10/25
#
# This script loads a FLAC audio file and adds a gap to it

# Import libraries
import librosa
import soundfile as sf
import numpy as np

from utils import load_audio
from config import SAMPLE_AUDIO_FILE

def insert_gap(audio_path, output_path, gap_start, gap_duration, sample_rate=16000):
    """
    Insert a gap into a FLAC audio file
    """
    # Load audio
    print(f"Loading audio...")
    y, orig_sr = load_audio(audio_path, sample_rate)
    
    # Convert time to sample index
    gap_start_idx = int(gap_start * sample_rate)
    gap_length = int(gap_duration * sample_rate)
    
    # Create silent gap
    silence = np.zeros(gap_length)
    
    # Split and insert silence
    print("Adding gap...")
    y_new = np.concatenate([y[:gap_start_idx], silence, y[gap_start_idx + gap_length:]])
    
    # Save new audio file
    print("Writing output file...")
    sf.write(output_path, y_new, sample_rate)
    
    print(f"Processed file saved to {output_path}")

# Example usage
if __name__ == "__main__":
    input_filepath    = SAMPLE_AUDIO_FILE
    output_filepath   = f"output/200-126784-0006_W_GAP.flac"
    gap_start_time    = 2.0     # Time in seconds where the gap starts
    gap_duration_time = 5.0     # Duration of the silence in seconds
    
    insert_gap(input_filepath, output_filepath, gap_start_time, gap_duration_time)
