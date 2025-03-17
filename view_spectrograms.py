# view_spectrograms.py
# CS 6140: Final Project - Audio Inpainting
# Written on 03/17/25
#
# This script loads a FLAC audio file and generates all forms of the spectrogram for it

import utils
from config import LIBRISPEECH_ROOT, LIBRISPEECH_ROOT_PROCESSED
import matplotlib.pyplot as plt

# Load sample processed audio file
audio_path_orig = LIBRISPEECH_ROOT / "103" / "1240" / "103-1240-0002.flac"
audio_path_proc = LIBRISPEECH_ROOT_PROCESSED / "103" / "1240" / "103-1240-0002.flac"
audio_data_orig, sr = utils.load_audio(audio_path_orig)
audio_data_proc, sr = utils.load_audio(audio_path_proc)

# Create spectrograms
spectrogram_orig = utils.extract_spectrogram(audio_data_orig)
spectrogram_proc = utils.extract_spectrogram(audio_data_proc)
mel_spectrogram_orig = utils.extract_mel_spectrogram(audio_data_orig)
mel_spectrogram_proc = utils.extract_mel_spectrogram(audio_data_proc)

# Plot the histogram
fig1 = utils.visualize_spectrogram(spectrogram_orig, title="Original Audio Spectrogram")
fig2 = utils.visualize_spectrogram(spectrogram_proc, title="Gap Audio Spectrogram")
fig3 = utils.visualize_spectrogram(mel_spectrogram_orig, title="Original Audio Mel Spectrogram")
fig4 = utils.visualize_spectrogram(mel_spectrogram_orig, title="Gap Audio Mel Spectrogram")

plt.show()