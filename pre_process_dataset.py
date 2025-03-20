# pre_process_dataset.py
# CS 6140 Final Project: Audio Inpainting
#
# This script pre-processes the LibriSpeech dataset
# 1. Load all the FLAC files
# 2. Add gaps at random position of length GAP_SIZE_MS 
# 3. Figure out whether to export new audio files, or store all data in a .npy/binary file - FOR NOW, recreate the LibriSpeech folder structure with new saved audio files
# 4. Save pre-processed files to the LIBRISPEECH_ROOT_PROCESSED directory
# NOTE: This script takes ~4.5 min to run

import os
import utils
from tqdm import tqdm
from pathlib import Path

# Load LibriSpeech paths
from config import LIBRISPEECH_ROOT, LIBRISPEECH_ROOT_PROCESSED, SUPPORTED_FORMATS

# Loop through all the files
file_count = sum(len(files) for _, _, files in os.walk(LIBRISPEECH_ROOT, topdown=True))
with tqdm(total=file_count, desc="Pre-Processing Dataset") as pbar:  # Do tqdm this way
    for root, subdirs, files in os.walk(LIBRISPEECH_ROOT, topdown=True):
        # Construct the destination path
        relative_path = os.path.relpath(root, LIBRISPEECH_ROOT)
        dest_path     = os.path.join(LIBRISPEECH_ROOT_PROCESSED, relative_path)
        os.makedirs(dest_path, exist_ok=True)

        # Make sure we are in a subdirectory with only audio files
        if (len(subdirs) == 0):
            for f in files:
                #print(f"Filepath: {Path(root) / Path(f)}")
                audio_path = Path(root) / Path(f)
                output_path = Path(dest_path) / Path(f)
                extension = audio_path.suffix

                # Ensure only FLAC files are processed
                if (extension in SUPPORTED_FORMATS):
                    audio_data_new, _  = utils.add_random_gap(audio_path, 0.1)
                    #print(f"Original path: {audio_path}")
                    #print(f"Destination path: {output_path}")
                    utils.save_audio(audio_data_new, output_path)

                    pbar.update(1)