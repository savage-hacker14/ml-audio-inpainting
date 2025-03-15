# pre_process_dataset.py
# CS 6140 Final Project: Audio Inpainting
#
# This script pre-processes the LibriSpeech dataset
# 1. Load all the FLAC files
# 2. Add gaps at random position of length GAP_SIZE_MS 
# 3. Figure out whether to export new audio files, or store all data in a .npy/binary file
# 4. Save pre-processed files to the LIBRISPEECH_ROOT_PROCESSED directory
