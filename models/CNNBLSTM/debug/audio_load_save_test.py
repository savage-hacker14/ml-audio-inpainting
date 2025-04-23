import librosa
import librosa.display
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

# Load audio file
file_path = "output/true_audio_0.flac"  # Change this to your audio file path
y, sr = librosa.load(file_path, sr=None)  # Load with original sample rate

# Compute STFT (magnitude + phase)
S = librosa.stft(y, n_fft=512, hop_length=192, window="hann", win_length=384)

# Convert to spectrogram (magnitude)
S_mag = np.abs(S)

# Plot spectrogram
plt.figure(figsize=(10, 4))
librosa.display.specshow(librosa.amplitude_to_db(S_mag, ref=np.max), sr=sr, hop_length=256, x_axis="time", y_axis="log")
plt.colorbar(label="dB")
plt.title("Spectrogram")
plt.show()

# Reconstruct audio using inverse STFT
y_reconstructed = librosa.istft(S, n_fft=512, hop_length=192, window="hann", win_length=384)

# Save reconstructed audio as a lossless WAV file
sf.write("output/true_audio_0_reconstructed.flac", y_reconstructed, sr)  # 24-bit WAV for lossless quality

print("Reconstruction complete! Saved as 'output.wav'.")
