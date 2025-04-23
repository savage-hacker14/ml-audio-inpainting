import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

def apply_gap_to_spectrogram(spectrogram: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Applies a gap to a spectrogram by zeroing out values based on a binary mask.

    Args:
        spectrogram (np.ndarray): The input spectrogram (2D or 3D array).
        mask (np.ndarray): A binary mask (same shape as spectrogram), where 1 indicates the gap.

    Returns:
        np.ndarray: The modified spectrogram with the gap applied.
    """
    if spectrogram.shape != mask.shape:
        raise ValueError("Spectrogram and mask must have the same shape")

    return spectrogram * (1 - mask)  # Zero out values where mask is 1


def plot_spectrogram_with_gap(S, S_gapped, sr, mask):
    """
    Plots the original and gapped spectrograms with vertical lines marking the gap.

    Args:
        S (np.ndarray): Original spectrogram.
        S_gapped (np.ndarray): Gapped spectrogram.
        sr (int): Sample rate.
        mask (np.ndarray): Binary mask indicating the gap.
    """
    # Find the time indices where the mask has gaps (assuming time is along axis 1)
    gap_columns = np.where(mask.any(axis=0))[0]  # Columns where mask has 1s
    if len(gap_columns) > 0:
        gap_start, gap_end = gap_columns[0], gap_columns[-1]
    else:
        gap_start, gap_end = None, None  # No gap detected

    plt.figure(figsize=(10, 5))

    # Plot original spectrogram
    plt.subplot(1, 2, 1)
    librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max), sr=sr, y_axis='log', x_axis='time')
    plt.title("Original Spectrogram")

    # Plot gapped spectrogram
    plt.subplot(1, 2, 2)
    librosa.display.specshow(librosa.amplitude_to_db(S_gapped, ref=np.max), sr=sr, y_axis='log', x_axis='time')
    plt.title("Gapped Spectrogram")

    # Draw vertical lines to mark the gap interval
    if gap_start is not None and gap_end is not None:
        times = librosa.frames_to_time([gap_start, gap_end], sr=sr)
        plt.axvline(x=times[0], color='r', linestyle='--', label="Gap Start")
        plt.axvline(x=times[1], color='r', linestyle='--', label="Gap End")
        #plt.legend()

    plt.show()


if __name__ == "__main__":
    # Example Usage
    y, sr = librosa.load(librosa.example("trumpet"))
    S = np.abs(librosa.stft(y, hop_length=512))

    # Create a binary mask (gap in a time range)
    mask = np.zeros_like(S)
    mask[:, 50:80] = 1  # Apply a vertical gap (across frequency)

    # Apply the gap
    S_gapped = apply_gap_to_spectrogram(S, mask)

    # Plot with vertical markers
    plot_spectrogram_with_gap(S, S_gapped, sr, mask)