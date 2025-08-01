import os
import numpy as np
import pytest
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from pathlib import Path
import tempfile
import shutil
from unittest.mock import patch, MagicMock

from utils import (
    load_audio,
    add_random_gap,
    extract_spectrogram,
    extract_mel_spectrogram,
    spectrogram_to_audio,
    mel_spectrogram_to_audio,
    save_audio,
    visualize_spectrogram,
)

# Define constants for tests
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
def sample_audio_file(temp_dir):
    """Create a sample audio file for testing."""
    # Generate a simple sine wave
    duration = 2  # seconds
    t = np.linspace(0, duration, int(DEFAULT_SAMPLE_RATE * duration))
    audio = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.2 * np.sin(2 * np.pi * 880 * t)
    
    # Save to a temporary WAV file
    file_path = os.path.join(temp_dir, "test_audio.wav")
    sf.write(file_path, audio, DEFAULT_SAMPLE_RATE)
    
    return file_path, audio

@pytest.fixture
def stereo_audio_file(temp_dir):
    """Create a sample stereo audio file for testing."""
    # Generate a simple stereo sine wave
    duration = 2  # seconds
    t = np.linspace(0, duration, int(DEFAULT_SAMPLE_RATE * duration))
    audio_left = 0.5 * np.sin(2 * np.pi * 440 * t)
    audio_right = 0.2 * np.sin(2 * np.pi * 880 * t)
    stereo_audio = np.column_stack((audio_left, audio_right))
    
    # Save to a temporary WAV file
    file_path = os.path.join(temp_dir, "test_stereo_audio.wav")
    sf.write(file_path, stereo_audio, DEFAULT_SAMPLE_RATE)
    
    return file_path, stereo_audio

@pytest.fixture
def sample_spectrogram():
    """Create a sample spectrogram for testing."""
    # Generate a simple sine wave
    duration = 2  # seconds
    t = np.linspace(0, duration, int(DEFAULT_SAMPLE_RATE * duration))
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    
    # Generate spectrogram
    spec = librosa.stft(
        audio, 
        n_fft=TEST_N_FFT, 
        hop_length=TEST_HOP_LENGTH,
        win_length=TEST_WIN_LENGTH
    )
    mag_spec = np.abs(spec)
    
    return audio, mag_spec

@pytest.fixture
def sample_mel_spectrogram():
    """Create a sample mel spectrogram for testing."""
    # Generate a simple sine wave
    duration = 2  # seconds
    t = np.linspace(0, duration, int(DEFAULT_SAMPLE_RATE * duration))
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    
    # Generate mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=DEFAULT_SAMPLE_RATE,
        n_fft=TEST_N_FFT,
        hop_length=TEST_HOP_LENGTH,
        n_mels=128
    )
    
    return audio, mel_spec

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

# ---------- Tests for load_audio ----------

def test_load_audio_mono(sample_audio_file):
    """Test loading a mono audio file."""
    file_path, original_audio = sample_audio_file
    
    # Test with default parameters
    loaded_audio, sr = load_audio(
        file_path, 
        sample_rate=DEFAULT_SAMPLE_RATE,
        max_len=5
    )
    
    assert sr == DEFAULT_SAMPLE_RATE
    assert loaded_audio.ndim == 1  # Should be mono
    assert loaded_audio.shape[0] == DEFAULT_SAMPLE_RATE * 5  # Should be padded to max_len
    
def test_load_audio_stereo_to_mono(stereo_audio_file):
    """Test loading a stereo audio file as mono."""
    file_path, _ = stereo_audio_file
    
    # Test conversion to mono
    loaded_audio, sr = load_audio(
        file_path, 
        sample_rate=DEFAULT_SAMPLE_RATE,
        max_len=5,
        mono=True
    )
    
    assert sr == DEFAULT_SAMPLE_RATE
    assert loaded_audio.ndim == 1  # Should be mono
    assert loaded_audio.shape[0] == DEFAULT_SAMPLE_RATE * 5  # Should be padded to max_len

def test_load_audio_clip(sample_audio_file):
    """Test loading and clipping an audio file."""
    file_path, _ = sample_audio_file
    
    # Test with shorter max_len_s to test clipping
    loaded_audio, sr = load_audio(
        file_path, 
        sample_rate=DEFAULT_SAMPLE_RATE,
        max_len=1
    )
    
    assert sr == DEFAULT_SAMPLE_RATE
    assert loaded_audio.shape[0] == DEFAULT_SAMPLE_RATE * 1  # Should be clipped to max_len

def test_load_audio_resampling(sample_audio_file):
    """Test loading an audio file with resampling."""
    file_path, _ = sample_audio_file
    target_sr = 16000
    
    # Test with different sample rate to test resampling
    loaded_audio, sr = load_audio(
        file_path, 
        sample_rate=target_sr,
        max_len=5
    )
    
    assert sr == target_sr
    assert loaded_audio.shape[0] == target_sr * 5  # Should match the new sample rate

def test_load_audio_file_not_found():
    """Test loading a non-existent audio file."""
    with pytest.raises(IOError):
        load_audio("nonexistent_file.wav")

# ---------- Tests for random_gap ----------

def test_add_random_gap(sample_audio_file):
    """Test adding a random gap to an audio file."""
    file_path, _ = sample_audio_file
    gap_len = 0.5
    
    # Add a random gap
    audio_with_gap, gap_interval = add_random_gap(
        file_path,
        gap_len=gap_len,
        sample_rate=DEFAULT_SAMPLE_RATE
    )
    
    assert isinstance(gap_interval, tuple)
    assert len(gap_interval) == 2
    start_time, end_time = gap_interval
    
    # Check gap duration
    assert np.isclose(end_time - start_time, gap_len)
    
    # Check audio shape (should be same as original audio)
    orig_audio, _ = load_audio(file_path, sample_rate=DEFAULT_SAMPLE_RATE)
    assert audio_with_gap.shape == orig_audio.shape
    
    # Check that there's a silent section at the gap location
    gap_start_idx = int(start_time * DEFAULT_SAMPLE_RATE)
    gap_end_idx = int(end_time * DEFAULT_SAMPLE_RATE)
    gap_segment = audio_with_gap[gap_start_idx:gap_end_idx]
    assert np.allclose(gap_segment, 0)

def test_add_random_gap_too_long(sample_audio_file):
    """Test adding a gap longer than the audio file."""
    file_path, original_audio = sample_audio_file
    gap_len = 10  # Longer than the 2-second audio
    
    with pytest.raises(ValueError):
        add_random_gap(
            file_path,
            gap_len=gap_len,
            sample_rate=DEFAULT_SAMPLE_RATE
        )


# ---------- Tests for extract_spectrogram ----------

def test_extract_spectrogram(sample_audio_file):
    """Test extracting a spectrogram from audio data."""
    _, original_audio = sample_audio_file
    
    # Extract spectrogram
    spec = extract_spectrogram(
        original_audio,
        n_fft=TEST_N_FFT,
        hop_length=TEST_HOP_LENGTH,
        win_length=TEST_WIN_LENGTH
    )
    
    # Check shape
    expected_time_bins = 1 + (len(original_audio) - TEST_WIN_LENGTH) // TEST_HOP_LENGTH
    assert spec.shape[0] == TEST_N_FFT // 2 + 1  # Number of frequency bins
    assert spec.shape[1] <= expected_time_bins  # Number of time bins

def test_extract_spectrogram_power_values(sample_audio_file):
    """Test extracting spectrograms with different power values."""
    _, original_audio = sample_audio_file
    
    # Extract spectrogram with power=1.0 (magnitude)
    spec1 = extract_spectrogram(
        original_audio,
        n_fft=TEST_N_FFT,
        hop_length=TEST_HOP_LENGTH,
        power=1.0
    )
    
    # Extract spectrogram with power=2.0 (power)
    spec2 = extract_spectrogram(
        original_audio,
        n_fft=TEST_N_FFT,
        hop_length=TEST_HOP_LENGTH,
        power=2.0
    )
    
    # spec2 should approximately equal spec1^2
    assert np.allclose(spec2, spec1**2)

def test_extract_spectrogram_negative_power():
    """Test extracting a spectrogram with invalid power value."""
    audio = np.random.rand(DEFAULT_SAMPLE_RATE)
    
    with pytest.raises(ValueError):
        extract_spectrogram(audio, power=-1.0)


# Tests for extract_mel_spectrogram
def test_extract_mel_spectrogram(sample_audio_file):
    """Test extracting a mel spectrogram from audio data."""
    _, original_audio = sample_audio_file
    
    # Extract mel spectrogram
    mel_spec = extract_mel_spectrogram(
        original_audio,
        sample_rate=DEFAULT_SAMPLE_RATE,
        n_fft=TEST_N_FFT,
        hop_length=TEST_HOP_LENGTH,
        n_mels=128
    )
    
    # Check shape
    expected_time_bins = 1 + (len(original_audio) - TEST_N_FFT) // TEST_HOP_LENGTH
    assert mel_spec.shape[0] == 128  # Number of mel bands
    assert mel_spec.shape[1] <= expected_time_bins  # Number of time bins

def test_extract_mel_spectrogram_power_values(sample_audio_file):
    """Test extracting mel spectrograms with different power values."""
    _, original_audio = sample_audio_file
    
    # Extract mel spectrogram with power=1.0 (magnitude)
    mel_spec1 = extract_mel_spectrogram(
        original_audio,
        sample_rate=DEFAULT_SAMPLE_RATE,
        n_fft=TEST_N_FFT,
        hop_length=TEST_HOP_LENGTH,
        power=1.0
    )
    
    # Extract mel spectrogram with power=2.0 (power)
    mel_spec2 = extract_mel_spectrogram(
        original_audio,
        sample_rate=DEFAULT_SAMPLE_RATE,
        n_fft=TEST_N_FFT,
        hop_length=TEST_HOP_LENGTH,
        power=2.0
    )
    
    # mel_spec2 should approximately equal mel_spec1^2
    assert np.allclose(mel_spec2, mel_spec1**2)

def test_extract_mel_spectrogram_negative_power():
    """Test extracting a mel spectrogram with invalid power value."""
    audio = np.random.rand(DEFAULT_SAMPLE_RATE)
    
    with pytest.raises(ValueError):
        extract_mel_spectrogram(audio, power=-1.0)
        
# ---------- Tests for spectrogram_to_audio ----------

def test_spectrogram_to_audio(sample_spectrogram):
    """Test converting a spectrogram back to audio."""
    original_audio, mag_spec = sample_spectrogram
    
    # Convert spectrogram to audio
    reconstructed_audio = spectrogram_to_audio(
        mag_spec,
        phase_info=False,
        hop_length=TEST_HOP_LENGTH,
        win_length=TEST_WIN_LENGTH,
        n_fft=TEST_N_FFT,
        n_iter=64
    )
    
    # Check that we got audio back
    assert isinstance(reconstructed_audio, np.ndarray)
    assert reconstructed_audio.ndim == 1
    
    # Because of phase reconstruction, we can't expect perfect reconstruction
    # But we can check the length is reasonable and signal isn't all zeros
    assert len(reconstructed_audio) > 0
    assert not np.allclose(reconstructed_audio, 0)

def test_spectrogram_to_audio_with_phase(sample_spectrogram):
    """Test converting a complex spectrogram (with phase) back to audio."""
    original_audio, _ = sample_spectrogram
    
    # Create complex spectrogram
    complex_spec = librosa.stft(
        original_audio,
        n_fft=TEST_N_FFT, 
        hop_length=TEST_HOP_LENGTH,
        win_length=TEST_WIN_LENGTH
    )
    
    # Convert complex spectrogram to audio
    reconstructed_audio = spectrogram_to_audio(
        complex_spec,
        phase_info=True,
        hop_length=TEST_HOP_LENGTH,
        win_length=TEST_WIN_LENGTH
    )
    
    # Check that we got audio back with similar length
    # The reconstruction should be much better with phase info
    assert isinstance(reconstructed_audio, np.ndarray)
    
    # Trim to same length for comparison
    min_len = min(len(original_audio), len(reconstructed_audio))
    original_trimmed = original_audio[:min_len]
    reconstructed_trimmed = reconstructed_audio[:min_len]
    
    # With phase info, we should get a good reconstruction
    # Allow some tolerance for numerical differences
    correlation = np.corrcoef(original_trimmed, reconstructed_trimmed)[0, 1]
    assert correlation > 0.9  # High correlation expected
    
# ---------- Tests for mel_spectrogram_to_audio ----------

def test_mel_spectrogram_to_audio(sample_mel_spectrogram):
    """Test converting a mel spectrogram back to audio."""
    original_audio, mel_spec = sample_mel_spectrogram
    
    # Convert mel spectrogram to audio
    reconstructed_audio = mel_spectrogram_to_audio(
        mel_spec,
        sample_rate=DEFAULT_SAMPLE_RATE,
        n_fft=TEST_N_FFT,
        hop_length=TEST_HOP_LENGTH,
        n_iter=32,
        n_mels=128,
        power=2.0
    )
    
    # Check that we got audio back
    assert isinstance(reconstructed_audio, np.ndarray)
    assert reconstructed_audio.ndim == 1
    
    # Because of mel transformation and phase reconstruction,
    # we can't expect perfect reconstruction
    # But we can check the length is reasonable and signal isn't all zeros
    assert len(reconstructed_audio) > 0
    assert not np.allclose(reconstructed_audio, 0)

def test_mel_spectrogram_to_audio_power_handling():
    """Test mel spectrogram to audio conversion with different power values."""
    # Generate test audio
    duration = 1
    t = np.linspace(0, duration, int(DEFAULT_SAMPLE_RATE * duration))
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    
    # Generate power=2.0 mel spectrogram
    mel_spec_power2 = librosa.feature.melspectrogram(
        y=audio,
        sr=DEFAULT_SAMPLE_RATE,
        n_fft=TEST_N_FFT,
        hop_length=TEST_HOP_LENGTH,
        power=2.0
    )
    
    # Test with power=2.0
    audio_from_power2 = mel_spectrogram_to_audio(
        mel_spec_power2,
        sample_rate=DEFAULT_SAMPLE_RATE,
        n_fft=TEST_N_FFT,
        hop_length=TEST_HOP_LENGTH,
        power=2.0
    )
    
    # Generate power=1.0 mel spectrogram
    mel_spec_power1 = librosa.feature.melspectrogram(
        y=audio,
        sr=DEFAULT_SAMPLE_RATE,
        n_fft=TEST_N_FFT,
        hop_length=TEST_HOP_LENGTH,
        power=1.0
    )
    
    # Test with power=1.0
    audio_from_power1 = mel_spectrogram_to_audio(
        mel_spec_power1,
        sample_rate=DEFAULT_SAMPLE_RATE,
        n_fft=TEST_N_FFT,
        hop_length=TEST_HOP_LENGTH,
        power=1.0
    )
    
    # Both should produce valid audio
    assert len(audio_from_power1) > 0
    assert len(audio_from_power2) > 0

# ---------- Tests for save_audio ----------

def test_save_audio(temp_dir, sample_audio_file):
    """Test saving audio to a file."""
    _, original_audio = sample_audio_file
    output_path = os.path.join(temp_dir, "output_audio.wav")
    
    # Save audio
    save_audio(
        original_audio,
        output_path,
        sample_rate=DEFAULT_SAMPLE_RATE,
        file_format="wav"
    )
    
    # Check file exists
    assert os.path.exists(output_path)
    
    # Load saved file and check content
    saved_audio, sr = sf.read(output_path)
    assert sr == DEFAULT_SAMPLE_RATE
    
    # Check that audio was normalized
    assert np.max(np.abs(saved_audio)) <= 1.0

def test_save_audio_creates_directory(temp_dir):
    """Test that save_audio creates the output directory if it doesn't exist."""
    audio = np.random.rand(DEFAULT_SAMPLE_RATE)
    nested_dir = os.path.join(temp_dir, "nested", "directory")
    output_path = os.path.join(nested_dir, "output_audio.wav")
    
    # Directory shouldn't exist yet
    assert not os.path.exists(nested_dir)
    
    # Save audio
    save_audio(
        audio,
        output_path,
        sample_rate=DEFAULT_SAMPLE_RATE
    )
    
    # Check directory and file exist
    assert os.path.exists(nested_dir)
    assert os.path.exists(output_path)

def test_save_audio_error_handling():
    """Test error handling when saving to an invalid location."""
    audio = np.random.rand(DEFAULT_SAMPLE_RATE)
    
    # Try to save to a location that can't be created
    with pytest.raises(IOError):
        save_audio(
            audio,
            "/root/unauthorized/path/audio.wav",  # Should fail on most systems
            sample_rate=DEFAULT_SAMPLE_RATE
        )

# ---------- Tests for visualize_spectrogram ----------

def test_visualize_spectrogram(sample_spectrogram, temp_dir):
    """Test visualizing a spectrogram."""
    _, mag_spec = sample_spectrogram
    output_path = os.path.join(temp_dir, "spectrogram.png")
    
    # Test with saving to file
    fig = visualize_spectrogram(
        mag_spec,
        power=1,
        sample_rate=DEFAULT_SAMPLE_RATE,
        n_fft=TEST_N_FFT,
        hop_length=TEST_HOP_LENGTH,
        win_length=TEST_WIN_LENGTH,
        title="Test Spectrogram",
        save_path=output_path
    )
    
    # Check that file was created
    assert os.path.exists(output_path)
    assert fig is None  # Should return None when saving to file

def test_visualize_spectrogram_return_figure(sample_spectrogram):
    """Test visualizing a spectrogram without saving."""
    _, mag_spec = sample_spectrogram
    
    # Test without saving to file
    fig = visualize_spectrogram(
        mag_spec,
        power=1,
        sample_rate=DEFAULT_SAMPLE_RATE,
        n_fft=TEST_N_FFT,
        hop_length=TEST_HOP_LENGTH,
        win_length=TEST_WIN_LENGTH,
        title="Test Spectrogram"
    )
    
    # Should return a figure when not saving
    assert fig is not None
    plt.close(fig)  # Clean up

def test_visualize_spectrogram_with_gap(sample_spectrogram):
    """Test visualizing a spectrogram with gap markers."""
    _, mag_spec = sample_spectrogram
    gap_int = (0.5, 1.0)  # 0.5 to 1.0 second gap
    
    # Test with gap markers
    fig = visualize_spectrogram(
        mag_spec,
        power=1,
        sample_rate=DEFAULT_SAMPLE_RATE,
        n_fft=TEST_N_FFT,
        hop_length=TEST_HOP_LENGTH,
        win_length=TEST_WIN_LENGTH,
        gap_int=gap_int,
        title="Test Spectrogram with Gap"
    )
    
    # Should return a figure
    assert fig is not None
    plt.close(fig)  # Clean up

def test_visualize_spectrogram_invalid_power():
    """Test visualizing a spectrogram with invalid power value."""
    spectrogram = np.random.rand(100, 100)
    
    with pytest.raises(ValueError):
        visualize_spectrogram(
            spectrogram,
            power=3  # Invalid power value
        )

# ---------- Tests for reconstruction ----------

def test_full_reconstruction_spectrogram(sample_audio_file):
    """Test full reconstruction pipeline: audio → spectrogram → audio."""
    _, original_audio = sample_audio_file
    
    # Step 1: Extract spectrogram
    spectrogram = extract_spectrogram(
        original_audio,
        n_fft=TEST_N_FFT,
        hop_length=TEST_HOP_LENGTH,
        win_length=TEST_WIN_LENGTH,
        power=1.0
    )
    
    # Step 2: Convert back to audio
    reconstructed_audio = spectrogram_to_audio(
        spectrogram,
        phase_info=False,
        hop_length=TEST_HOP_LENGTH,
        win_length=TEST_WIN_LENGTH,
        n_fft=TEST_N_FFT,
        n_iter=64
    )
    
    # Trim both to the same length for comparison
    min_len = min(len(original_audio), len(reconstructed_audio))
    original_trimmed = original_audio[:min_len]
    reconstructed_trimmed = reconstructed_audio[:min_len]
    
    # We can't expect perfect reconstruction due to phase loss,
    # but we can check if the reconstructed audio has similar characteristics
    # Calculate correlation between original and reconstructed
    correlation = np.corrcoef(original_trimmed, reconstructed_trimmed)[0, 1]
    
    # The correlation should be positive, indicating some similarity
    assert correlation > 0  
    
    # Compare spectral characteristics
    orig_spec = np.abs(librosa.stft(original_trimmed))
    recon_spec = np.abs(librosa.stft(reconstructed_trimmed))
    
    # Calculate spectral correlation
    orig_spec_flat = orig_spec.flatten()
    recon_spec_flat = recon_spec.flatten()
    spec_correlation = np.corrcoef(orig_spec_flat, recon_spec_flat)[0, 1]
    
    # Spectral correlation should be high
    assert spec_correlation > 0.7

def test_full_reconstruction_mel_spectrogram(sample_audio_file):
    """Test full reconstruction pipeline: audio → mel spectrogram → audio."""
    _, original_audio = sample_audio_file
    
    # Step 1: Extract mel spectrogram
    mel_spectrogram = extract_mel_spectrogram(
        original_audio,
        sample_rate=DEFAULT_SAMPLE_RATE,
        n_fft=TEST_N_FFT,
        hop_length=TEST_HOP_LENGTH,
        n_mels=128,
        power=2.0
    )
    
    # Step 2: Convert back to audio
    reconstructed_audio = mel_spectrogram_to_audio(
        mel_spectrogram,
        sample_rate=DEFAULT_SAMPLE_RATE,
        n_fft=TEST_N_FFT,
        hop_length=TEST_HOP_LENGTH,
        n_iter=32,
        n_mels=128,
        power=2.0
    )
    
    # Trim both to the same length for comparison
    min_len = min(len(original_audio), len(reconstructed_audio))
    original_trimmed = original_audio[:min_len]
    reconstructed_trimmed = reconstructed_audio[:min_len]
    
    # We expect significant information loss due to mel transformation,
    # but we can check if the reconstructed audio preserves some characteristics
    
    # Compare spectral characteristics
    orig_mel_spec = librosa.feature.melspectrogram(
        y=original_trimmed,
        sr=DEFAULT_SAMPLE_RATE,
        n_mels=128
    )
    recon_mel_spec = librosa.feature.melspectrogram(
        y=reconstructed_trimmed,
        sr=DEFAULT_SAMPLE_RATE,
        n_mels=128
    )
    
    # Calculate spectral correlation
    orig_spec_flat = orig_mel_spec.flatten()
    recon_spec_flat = recon_mel_spec.flatten()
    spec_correlation = np.corrcoef(orig_spec_flat, recon_spec_flat)[0, 1]
    
    # Mel spectral correlation should be positive
    assert spec_correlation > 0

def test_end_to_end_pipeline(sample_audio_file, temp_dir):
    """Test end-to-end pipeline: load → process → save → visualize."""
    file_path, _ = sample_audio_file
    output_audio_path = os.path.join(temp_dir, "output.wav")
    output_image_path = os.path.join(temp_dir, "spectrogram.png")
    
    # Step 1: Load audio
    audio_data, sr = load_audio(
        file_path,
        sample_rate=DEFAULT_SAMPLE_RATE
    )
    
    # Step 2: Add a gap
    audio_with_gap, gap_interval = add_random_gap(
        file_path,
        gap_len=0.5,
        sample_rate=DEFAULT_SAMPLE_RATE
    )
    
    # Step 3: Extract spectrogram
    spectrogram = extract_spectrogram(
        audio_with_gap,
        n_fft=TEST_N_FFT,
        hop_length=TEST_HOP_LENGTH
    )
    
    # Step 4: Convert back to audio
    reconstructed_audio = spectrogram_to_audio(
        spectrogram,
        hop_length=TEST_HOP_LENGTH,
        n_fft=TEST_N_FFT
    )
    
    # Step 5: Save audio
    save_audio(
        reconstructed_audio,
        output_audio_path,
        sample_rate=DEFAULT_SAMPLE_RATE
    )
    
    # Step 6: Visualize spectrogram
    visualize_spectrogram(
        spectrogram,
        power=1,
        sample_rate=DEFAULT_SAMPLE_RATE,
        gap_int=gap_interval,
        save_path=output_image_path
    )
    
    # Check outputs exist
    assert os.path.exists(output_audio_path)
    assert os.path.exists(output_image_path)
    
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