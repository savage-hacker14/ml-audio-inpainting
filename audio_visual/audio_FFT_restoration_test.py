import sys
sys.path.append("..")
import utils

# Define a sample sound file
from config import SAMPLE_AUDIO_FILE, DEFAULT_SAMPLE_RATE

# NEW FFT SETTINGS
n_fft = 2048
win_len = 512
hop_len = 64

# Load the STFT data
audio_data, _ = utils.load_audio(SAMPLE_AUDIO_FILE, sample_rate=DEFAULT_SAMPLE_RATE)
spectrogram = utils.extract_spectrogram(audio_data, n_fft=n_fft, win_length=win_len, hop_length=hop_len)

# Export the spectrogram to a flac file
audio_recovered = utils.spectrogram_to_audio(spectrogram, phase_info=True, n_fft=n_fft, win_length=win_len, hop_length=hop_len)
utils.save_audio(audio_recovered, f"output/fft_test_nfft_{n_fft}_winlen_{win_len}_hoplen_{hop_len}.flac", sample_rate=DEFAULT_SAMPLE_RATE)


# ORIGINAL FFT SETTINGS
n_fft = 2048
win_len = 160
hop_len = 80

# Load the STFT data
audio_recovered = utils.spectrogram_to_audio(spectrogram, phase_info=True, n_fft=n_fft, win_length=win_len, hop_length=hop_len)
spectrogram = utils.extract_spectrogram(audio_data, n_fft=n_fft, win_length=win_len, hop_length=hop_len)

# Export the spectrogram to a flac file
audio_recovered = utils.spectrogram_to_audio(spectrogram, phase_info=True, n_fft=n_fft, win_length=win_len, hop_length=hop_len)
utils.save_audio(audio_recovered, f"output/fft_test_nfft_{n_fft}_winlen_{win_len}_hoplen_{hop_len}.flac", sample_rate=DEFAULT_SAMPLE_RATE)