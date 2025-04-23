import os
from pathlib import Path

# Base project directory - adjust this to your project location
PROJECT_ROOT = Path(os.path.dirname(os.path.abspath(__file__)))

# Define user (for local paths to LibriSpeech data)
USER = "Jacob"

# Dataset paths
if (USER == "Jacob"):
    LIBRISPEECH_ROOT = Path("C:\\Users\\Jacob\\Documents\\2024\\Northeastern\\CS_6140\\Project\\LibriSpeech\\train-clean-100")
    LIBRISPEECH_ROOT_PROCESSED = Path("C:\\Users\\Jacob\\Documents\\2024\\Northeastern\\CS_6140\\Project\\LibriSpeech_PROCESSED\\train-clean-100")
else:
    LIBRISPEECH_ROOT = Path("/LibriSpeech/train-clean-100")
    LIBRISPEECH_ROOT_PROCESSED = Path("/LibriSpeech_PROCESSED/train-clean-100")

# Sample audio file paths
SAMPLE_AUDIO_DIR = LIBRISPEECH_ROOT / "200/126784"
SAMPLE_AUDIO_FILE = SAMPLE_AUDIO_DIR / "200-126784-0006.flac"

# Output paths
OUTPUT_DIR = PROJECT_ROOT / "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)  # Create output directory if it doesn't exist

# Default parameters for audio/visual model
DEFAULT_SAMPLE_RATE = 16000  # 16 kHz
DEFAULT_N_FFT = 512  # Number of FFT points
DEFAULT_HANN_WINDOW_SIZE = 384 # 384 samples for 24 ms at 16 kHz
DEFAULT_HANN_HOP_LENGTH = 192  # 12 ms

DEFAULT_GAP_START_TIME = 2.0
DEFAULT_GAP_DURATION = 0.5

# Processing options
SUPPORTED_FORMATS = [".flac", ".wav", ".mp3"]