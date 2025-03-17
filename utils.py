import os
import numpy as np
import librosa
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Optional, Union

from config import DEFAULT_SAMPLE_RATE

def load_audio(
    file_path: Union[str, Path],
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    max_len_s: int = 5,
    mono: bool = True
) -> Tuple[np.ndarray, int]:
    """
    Load an audio file into a numpy array.

    Parameters
    ----------
    file_path (str or Path): Path to the audio file
    sample_rate (int, optional): Target sample rate
    mono (bool, optional): Whether to convert audio to mono

    Returns
    -------
    tuple
        (audio_data, sample_rate)
    """
    audio_data, sr = librosa.load(file_path, sr=sample_rate, mono=mono)

    # Clip audio to max_len_s
    if (len(audio_data) > sample_rate * max_len_s):
        audio_data = audio_data[:sample_rate * max_len_s]

    return audio_data, sr

def add_random_gap(
        file_path: Union[str, Path],
        gap_len_s: int,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        mono: bool = True
) -> Tuple[np.ndarray, int]:
    """
    Add a random gap of length gap_len_s at a random valid position within the audio file and return the audio data
    
    Parameters
    ----------
    file_path (str or Path): Path to the audio file
    gap_len_s (int): Gap length [s] to add at one location within the audio file
    sample_rate (int, optional): Target sample rate
    mono (bool, optional): Whether to convert audio to mono

    Returns
    -------
    tuple
        (audio_data, sample_rate)
    """
    audio_data, sr = load_audio(file_path, sample_rate=sample_rate)
    
    # Get sample indices
    audio_len     = len(audio_data)
    gap_start_idx = np.random.randint(0, audio_len - int(gap_len_s * sample_rate))
    gap_length    = int(gap_len_s * sample_rate)
    silence       = np.zeros(gap_length)

    # Add gap
    audio_new = np.concatenate([audio_data[:gap_start_idx], silence, audio_data[gap_start_idx + gap_length:]])

    return audio_new, sr

def extract_spectrogram(
    audio_data: np.ndarray,
    n_fft: int = 2048,
    hop_length: int = 512,
    win_length: Optional[int] = None,
    window: str = 'hann',
    center: bool = True,
    power: float = 2.0
) -> np.ndarray:
    """
    Extract magnitude spectrogram from audio data.

    Parameters
    ----------
    audio_data (np.ndarray): Audio time series
    n_fft (int, optional): FFT window size
    hop_length (int, optional): Number of samples between successive frames
    win_length (int or None, optional): Window length. If None, defaults to n_fft
    window (str, optional): Window specification
    center (bool, optional): If True, pad signal on both sides
    power (float, optional): Exponent for the magnitude spectrogram (e.g. 1 for energy, 2 for power)
    
    Returns
    -------
    np.ndarray
        Magnitude spectrogram
    """
    stft = librosa.stft(
        audio_data,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center
    )
    return np.abs(stft) ** power


def extract_mel_spectrogram(
    audio_data: np.ndarray,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_mels: int = 128,
    fmin: float = 0.0,
    fmax: Optional[float] = None,
    power: float = 2.0
) -> np.ndarray:
    """
    Extract mel spectrogram from audio data.

    Parameters
    ----------
    audio_data (np.ndarray): Audio time series
    sample_rate (int, optional): Sample rate of audio
    n_fft (int, optional): FFT window size
    hop_length (int, optional): Number of samples between successive frames
    n_mels (int, optional): Number of mel bands
    fmin (float, optional): Minimum frequency
    fmax (float or None, optional): Maximum frequency. If None, use sample_rate/2
    power (float, optional): Exponent for the magnitude spectrogram (e.g. 1 for energy, 2 for power)

    Returns
    -------
    np.ndarray
        Mel spectrogram
    """
    return librosa.feature.melspectrogram(
        y=audio_data,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        power=power
    )


def spectrogram_to_audio(
    spectrogram: np.ndarray,
    hop_length: int = 512,
    win_length: Optional[int] = None,
    n_fft: int = 2048,
    n_iter: int = 32,
    window: str = 'hann',
    center: bool = True,
    phase_initialization: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Convert a magnitude spectrogram to audio using Griffin-Lim algorithm.

    Parameters
    ----------
    spectrogram (np.ndarray): Magnitude spectrogram
    hop_length (int, optional): Number of samples between successive frames
    win_length (int or None, optional): Window length. If None, defaults to n_fft
    n_fft (int, optional): FFT window size
    n_iter (int, optional): Number of iterations for Griffin-Lim
    window (str, optional): Window specification
    center (bool, optional): If True, pad signal on both sides
    phase_initialization (np.ndarray or None, optional): Initial phase for Griffin-Lim
    
    Returns
    -------
    np.ndarray
        Audio time series
    """
    if spectrogram.min() >= 0:
        spectrogram = np.sqrt(spectrogram)

    # Use Griffin-Lim algorithm to recover phase
    audio_data = librosa.griffinlim(
        spectrogram,
        hop_length=hop_length,
        win_length=win_length,
        n_fft=n_fft,
        n_iter=n_iter,
        window=window,
        center=center,
        init=phase_initialization
    )
    
    return audio_data


def mel_spectrogram_to_audio(
    mel_spectrogram: np.ndarray,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_iter: int = 32,
    n_mels: int = 128,
    fmin: float = 0.0,
    fmax: Optional[float] = None,
    power: float = 2.0
) -> np.ndarray:
    """
    Convert a mel spectrogram to audio using inverse transformation and Griffin-Lim.

    Parameters
    ----------
    mel_spectrogram (np.ndarray): Mel spectrogram
    sample_rate (int, optional): Sample rate of audio
    n_fft (int, optional): FFT window size
    hop_length (int, optional): Number of samples between successive frames
    n_iter (int, optional): Number of iterations for Griffin-Lim
    n_mels (int, optional): Number of mel bands
    fmin (float, optional): Minimum frequency
    fmax (float or None, optional): Maximum frequency. If None, use sample_rate/2
    power (float, optional): Exponent for the magnitude spectrogram (e.g. 1 for energy, 2 for power)

    Returns
    -------
    np.ndarray
        Audio time series
    """
    # Create a mel filterbank
    mel_basis = librosa.filters.mel(
        sr=sample_rate, 
        n_fft=n_fft,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax
    )
    
    # Apply the pseudo-inverse of the mel filterbank
    magnitude_spectrogram = np.dot(mel_basis.T, mel_spectrogram)
    
    # If the input was a power spectrogram, take the square root
    if power == 2.0:
        magnitude_spectrogram = np.sqrt(magnitude_spectrogram)
    
    # Perform Griffin-Lim to recover the phase and convert to audio
    audio_data = librosa.griffinlim(
        magnitude_spectrogram,
        hop_length=hop_length,
        n_fft=n_fft,
        n_iter=n_iter
    )
    
    return audio_data


def save_audio(
    audio_data: np.ndarray,
    file_path: Union[str, Path],
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    format: str = 'flac'
) -> None:
    """
    Save audio data to a file.

    Parameters
    ----------
    audio_data (np.ndarray): Audio time series
    file_path (str or Path): Path to save the audio file
    sample_rate (int, optional): Sample rate of audio
    format (str, optional): Audio file format
    
    Returns
    -------
    None
    """
    # Normalize audio before saving
    audio_data = librosa.util.normalize(audio_data)
    sf.write(file_path, audio_data, sample_rate, format=format)


def visualize_spectrogram(
    spectrogram: np.ndarray,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    hop_length: int = 512,
    y_axis: str = 'log',
    x_axis: str = 'time',
    title: str = 'Spectrogram',
    save_path: Optional[Union[str, Path]] = None
) -> None:
    """
    Visualize a spectrogram.

    Parameters
    ----------
    spectrogram (np.ndarray): Spectrogram to visualize
    sample_rate (int, optional): Sample rate of audio
    hop_length (int, optional): Number of samples between successive frames
    y_axis (str, optional): Scale for the y-axis ('linear', 'log', or 'mel')
    x_axis (str, optional): Scale for the x-axis ('time' or 'frames')
    title (str, optional): Title for the plot
    save_path (str or Path or None, optional): Path to save the visualization. If None, the plot is displayed.
    
    Returns
    -------
    None
    """
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(
        librosa.amplitude_to_db(spectrogram, ref=np.max),
        sr=sample_rate,
        hop_length=hop_length,
        y_axis=y_axis,
        x_axis=x_axis
    )
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()