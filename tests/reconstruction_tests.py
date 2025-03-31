import os
import numpy as np
import pytest
import librosa
import soundfile as sf
from pathlib import Path
import tempfile
import shutil

# Import functions from the module (adjust import path as needed)
from utils import (
    load_audio,
    extract_spectrogram,
    spectrogram_to_audio,
)

from config import DEFAULT_SAMPLE_RATE, DEFAULT_HANN_WINDOW_SIZE, DEFAULT_HANN_HOP_LENGTH, DEFAULT_N_FFT

TEST_HOP_LENGTH = DEFAULT_HANN_HOP_LENGTH
TEST_N_FFT = DEFAULT_N_FFT
TEST_WIN_LENGTH = DEFAULT_HANN_WINDOW_SIZE

# ---------- Fixtures for tests ----------

@pytest.fixture
def temp_dir():
    """Create a temporary directory for file operations."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def test_signals():
    """Generate test signals with known properties for more rigorous testing."""
    # Signal 1: Simple sine wave
    t = np.linspace(0, 2, 2 * DEFAULT_SAMPLE_RATE)
    sine_440hz = 0.5 * np.sin(2 * np.pi * 440 * t)
    
    # Signal 2: Combination of sine waves
    sine_combo = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.3 * np.sin(2 * np.pi * 880 * t)
    
    # Signal 3: Chirp (frequency sweep)
    chirp = librosa.chirp(
        fmin=20, 
        fmax=8000, 
        duration=2,
        sr=DEFAULT_SAMPLE_RATE
    )
    
    # Signal 4: Impulse train
    impulse = np.zeros(2 * DEFAULT_SAMPLE_RATE)
    impulse[::DEFAULT_SAMPLE_RATE//10] = 1.0
    
    # Signal 5: White noise
    noise = np.random.randn(2 * DEFAULT_SAMPLE_RATE) * 0.1
    
    return {
        'sine': sine_440hz,
        'sine_combo': sine_combo,
        'chirp': chirp,
        'impulse': impulse,
        'noise': noise
    }

# ---------- Fixtures for Reconstruction ----------

def test_stft_perfect_reconstruction():
    """
    Test STFT perfect reconstruction with phase information.
    
    This test verifies that with phase information preserved,
    we can achieve perfect reconstruction.
    """
    # Generate a test signal
    t = np.linspace(0, 2, 2 * DEFAULT_SAMPLE_RATE)
    original_audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    
    # Compute STFT
    stft = librosa.stft(
        original_audio,
        n_fft=TEST_N_FFT,
        hop_length=TEST_HOP_LENGTH,
        win_length=TEST_WIN_LENGTH
    )
    
    # Reconstruct using ISTFT
    reconstructed_audio = librosa.istft(
        stft,
        hop_length=TEST_HOP_LENGTH,
        win_length=TEST_WIN_LENGTH,
        length=len(original_audio)
    )
    
    # Check perfect reconstruction (within numerical precision)
    assert len(reconstructed_audio) == len(original_audio)
    assert np.allclose(original_audio, reconstructed_audio, atol=1e-10)

def test_magnitude_phase_combination():
    """
    Test that combining magnitude and phase gives the original STFT.
    
    This test verifies that we can separate and recombine the magnitude
    and phase components of the STFT.
    """
    # Generate a test signal
    t = np.linspace(0, 2, 2 * DEFAULT_SAMPLE_RATE)
    original_audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    
    # Compute STFT
    stft = librosa.stft(
        original_audio,
        n_fft=TEST_N_FFT,
        hop_length=TEST_HOP_LENGTH,
        win_length=TEST_WIN_LENGTH
    )
    
    # Separate magnitude and phase
    magnitude = np.abs(stft)
    phase = np.exp(1.0j * np.angle(stft))
    
    # Recombine magnitude and phase
    stft_recombined = magnitude * phase
    
    # Check recombination (within numerical precision)
    assert np.allclose(stft, stft_recombined, atol=1e-10)
    
    # Reconstruct using recombined STFT
    reconstructed_audio = librosa.istft(
        stft_recombined,
        hop_length=TEST_HOP_LENGTH,
        win_length=TEST_WIN_LENGTH,
        length=len(original_audio)
    )
    
    # Check perfect reconstruction (within numerical precision)
    assert np.allclose(original_audio, reconstructed_audio, atol=1e-10)

def test_griffin_lim_reconstruction_quality(test_signals):
    """
    Test the quality of Griffin-Lim reconstruction for different signal types.
    
    This test evaluates how well Griffin-Lim performs on different
    types of signals compared to perfect reconstruction.
    """
    results = {}
    
    for signal_name, original_audio in test_signals.items():
        # Step 1: Extract magnitude spectrogram
        spectrogram = extract_spectrogram(
            original_audio,
            n_fft=TEST_N_FFT,
            hop_length=TEST_HOP_LENGTH,
            win_length=TEST_WIN_LENGTH,
            power=1.0
        )
        
        # Step 2: Convert back to audio using Griffin-Lim
        reconstructed_audio = spectrogram_to_audio(
            spectrogram,
            phase_info=False,
            hop_length=TEST_HOP_LENGTH,
            win_length=TEST_WIN_LENGTH,
            n_fft=TEST_N_FFT,
            n_iter=100  # Use more iterations for better quality
        )
        
        # Trim to the same length for comparison
        min_len = min(len(original_audio), len(reconstructed_audio))
        original_trimmed = original_audio[:min_len]
        reconstructed_trimmed = reconstructed_audio[:min_len]
        
        # Calculate correlation
        correlation = np.corrcoef(original_trimmed, reconstructed_trimmed)[0, 1]
        results[signal_name] = correlation
        
        # Calculate spectral correlation
        orig_spec = np.abs(librosa.stft(original_trimmed))
        recon_spec = np.abs(librosa.stft(reconstructed_trimmed))
        
        orig_spec_flat = orig_spec.flatten()
        recon_spec_flat = recon_spec.flatten()
        spec_correlation = np.corrcoef(orig_spec_flat, recon_spec_flat)[0, 1]
        
        # For sine waves and chirps, reconstruction should be quite good
        if signal_name in ['sine', 'sine_combo', 'chirp']:
            assert spec_correlation > 0.9, f"Poor spectral correlation for {signal_name}"
        
        # Even for noise and impulses, spectral envelope should be preserved
        assert spec_correlation > 0.7, f"Very poor spectral correlation for {signal_name}"
    
    # Print results for reference
    print(f"Reconstruction quality by signal type: {results}")

def test_griffin_lim_convergence():
    """
    Test Griffin-Lim convergence with increasing iterations.
    
    This test verifies that more Griffin-Lim iterations lead to
    better reconstruction quality.
    """
    # Generate a test signal
    t = np.linspace(0, 2, 2 * DEFAULT_SAMPLE_RATE)
    original_audio = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.3 * np.sin(2 * np.pi * 880 * t)
    
    # Extract magnitude spectrogram
    spectrogram = extract_spectrogram(
        original_audio,
        n_fft=TEST_N_FFT,
        hop_length=TEST_HOP_LENGTH,
        win_length=TEST_WIN_LENGTH,
        power=1.0
    )
    
    correlations = []
    iterations = [10, 32, 64, 100]
    
    for n_iter in iterations:
        # Convert back to audio using Griffin-Lim with different iteration counts
        reconstructed_audio = spectrogram_to_audio(
            spectrogram,
            phase_info=False,
            hop_length=TEST_HOP_LENGTH,
            win_length=TEST_WIN_LENGTH,
            n_fft=TEST_N_FFT,
            n_iter=n_iter
        )
        
        # Trim to the same length for comparison
        min_len = min(len(original_audio), len(reconstructed_audio))
        original_trimmed = original_audio[:min_len]
        reconstructed_trimmed = reconstructed_audio[:min_len]
        
        # Calculate correlation
        correlation = np.corrcoef(original_trimmed, reconstructed_trimmed)[0, 1]
        correlations.append(correlation)
    
    # Verify that more iterations lead to better reconstruction
    assert correlations[0] <= correlations[-1], "More iterations did not improve quality"
    
    # Check for monotonic improvement (relaxed condition)
    # Note: In some cases, due to algorithm properties, monotonic improvement
    # might not be guaranteed for all iteration counts
    assert all(correlations[i] <= correlations[i+1] for i in range(len(correlations)-2))

def test_spectrogram_to_audio_with_phase_init():
    """
    Test spectrogram_to_audio with phase initialization.
    
    This test verifies that providing a phase initialization to Griffin-Lim
    works correctly and affects the reconstruction.
    """
    # Generate a test signal
    t = np.linspace(0, 2, 2 * DEFAULT_SAMPLE_RATE)
    original_audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    
    # Extract magnitude spectrogram
    spectrogram = extract_spectrogram(
        original_audio,
        n_fft=TEST_N_FFT,
        hop_length=TEST_HOP_LENGTH,
        win_length=TEST_WIN_LENGTH,
        power=1.0
    )
    
    # Create a random phase initialization
    random_phase = np.exp(1.0j * np.random.uniform(0, 2*np.pi, spectrogram.shape))
    
    # Reconstruct with random phase initialization
    reconstructed_audio_random = spectrogram_to_audio(
        spectrogram,
        phase_info=False,
        hop_length=TEST_HOP_LENGTH,
        win_length=TEST_WIN_LENGTH,
        n_fft=TEST_N_FFT,
        n_iter=32,
        phase_initialization=random_phase
    )
    
    # Reconstruct with default (zeros) phase initialization
    reconstructed_audio_default = spectrogram_to_audio(
        spectrogram,
        phase_info=False,
        hop_length=TEST_HOP_LENGTH,
        win_length=TEST_WIN_LENGTH,
        n_fft=TEST_N_FFT,
        n_iter=32,
        phase_initialization=None
    )
    
    # The reconstructions should be different (since we used different initializations)
    assert not np.allclose(reconstructed_audio_random, reconstructed_audio_default)

def test_stft_window_effects():
    """
    Test the effect of different window types on STFT reconstruction.
    
    This test evaluates how different window functions affect
    the quality of reconstruction.
    """
    # Generate a test signal
    t = np.linspace(0, 2, 2 * DEFAULT_SAMPLE_RATE)
    original_audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    
    windows = ['hann', 'hamming', 'blackman']
    correlations = []
    
    for window in windows:
        # Extract spectrogram with specific window
        spectrogram = extract_spectrogram(
            original_audio,
            n_fft=TEST_N_FFT,
            hop_length=TEST_HOP_LENGTH,
            win_length=TEST_WIN_LENGTH,
            window=window,
            power=1.0
        )
        
        # Convert back to audio
        reconstructed_audio = spectrogram_to_audio(
            spectrogram,
            phase_info=False,
            hop_length=TEST_HOP_LENGTH,
            win_length=TEST_WIN_LENGTH,
            n_fft=TEST_N_FFT,
            window=window,
            n_iter=64
        )
        
        # Trim to the same length for comparison
        min_len = min(len(original_audio), len(reconstructed_audio))
        original_trimmed = original_audio[:min_len]
        reconstructed_trimmed = reconstructed_audio[:min_len]
        
        # Calculate correlation
        correlation = np.corrcoef(original_trimmed, reconstructed_trimmed)[0, 1]
        correlations.append(correlation)
    
    # All window types should produce reasonable reconstructions
    assert all(corr > 0.5 for corr in correlations)
    
    # Print window comparisons
    for i, window in enumerate(windows):
        print(f"{window} window correlation: {correlations[i]}")

def test_hop_length_effects():
    """
    Test the effect of different hop lengths on STFT reconstruction.
    
    This test evaluates how different hop lengths affect
    the quality of reconstruction.
    """
    # Generate a test signal
    t = np.linspace(0, 2, 2 * DEFAULT_SAMPLE_RATE)
    original_audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    
    hop_lengths = [128, 256, 512, 1024]
    correlations = []
    
    for hop_length in hop_lengths:
        # Extract spectrogram with specific hop length
        spectrogram = extract_spectrogram(
            original_audio,
            n_fft=TEST_N_FFT,
            hop_length=hop_length,
            win_length=TEST_WIN_LENGTH,
            power=1.0
        )
        
        # Convert back to audio
        reconstructed_audio = spectrogram_to_audio(
            spectrogram,
            phase_info=False,
            hop_length=hop_length,
            win_length=TEST_WIN_LENGTH,
            n_fft=TEST_N_FFT,
            n_iter=64
        )
        
        # Trim to the same length for comparison
        min_len = min(len(original_audio), len(reconstructed_audio))
        original_trimmed = original_audio[:min_len]
        reconstructed_trimmed = reconstructed_audio[:min_len]
        
        # Calculate correlation
        correlation = np.corrcoef(original_trimmed, reconstructed_trimmed)[0, 1]
        correlations.append(correlation)
    
    # All hop lengths should produce reasonable reconstructions
    assert all(corr > 0.5 for corr in correlations)
    
    # Smaller hop lengths should generally give better reconstruction
    # (more overlap means better reconstruction)
    assert correlations[0] >= correlations[-1]
    
    # Print hop length comparisons
    for i, hop_length in enumerate(hop_lengths):
        print(f"Hop length {hop_length} correlation: {correlations[i]}")

def test_real_audio_file_reconstruction(temp_dir):
    """
    Test reconstruction with a real audio file.
    
    This test evaluates the full pipeline with a real audio file
    instead of synthetic signals.
    """
    # Create a sample audio file
    duration = 2  # seconds
    t = np.linspace(0, duration, int(DEFAULT_SAMPLE_RATE * duration))
    audio = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.2 * np.sin(2 * np.pi * 880 * t)
    
    file_path = os.path.join(temp_dir, "test_audio.wav")
    output_path = os.path.join(temp_dir, "reconstructed_audio.wav")
    
    # Save original audio
    sf.write(file_path, audio, DEFAULT_SAMPLE_RATE)
    
    # Full pipeline: load → spectrogram → audio → save
    # Step 1: Load audio
    audio_data, sr = load_audio(
        file_path,
        sample_rate=DEFAULT_SAMPLE_RATE
    )
    
    # Step 2: Extract spectrogram
    spectrogram = extract_spectrogram(
        audio_data,
        n_fft=TEST_N_FFT,
        hop_length=TEST_HOP_LENGTH,
        win_length=TEST_WIN_LENGTH,
        power=1.0
    )
    
    # Step 3: Convert back to audio
    reconstructed_audio = spectrogram_to_audio(
        spectrogram,
        phase_info=False,
        hop_length=TEST_HOP_LENGTH,
        win_length=TEST_WIN_LENGTH,
        n_fft=TEST_N_FFT,
        n_iter=64
    )
    
    # Step 4: Save reconstructed audio
    sf.write(output_path, reconstructed_audio, DEFAULT_SAMPLE_RATE)
    
    # Load the saved reconstructed audio
    reloaded_audio, sr = sf.read(output_path)
    
    # Trim to the same length for comparison
    min_len = min(len(audio), len(reloaded_audio))
    original_trimmed = audio[:min_len]
    reconstructed_trimmed = reloaded_audio[:min_len]
    
    # Calculate correlation
    correlation = np.corrcoef(original_trimmed, reconstructed_trimmed)[0, 1]
    
    # For a mix of sine waves, reconstruction should be reasonable
    assert correlation > 0.7
    
    # Calculate spectral correlation
    orig_spec = np.abs(librosa.stft(original_trimmed))
    recon_spec = np.abs(librosa.stft(reconstructed_trimmed))
    
    orig_spec_flat = orig_spec.flatten()
    recon_spec_flat = recon_spec.flatten()
    spec_correlation = np.corrcoef(orig_spec_flat, recon_spec_flat)[0, 1]
    
    # Spectral correlation should be high
    assert spec_correlation > 0.9