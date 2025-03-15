import os
from pathlib import Path

# Base project directory - adjust this to your project location
PROJECT_ROOT = Path(os.path.dirname(os.path.abspath(__file__)))

# Dataset paths
LIBRISPEECH_ROOT = Path("/home/jacob/Documents/2025/Northeastern/CS_6140/Audio_Inpainting_Project/LibriSpeech/train-clean-100/LibriSpeech/train-clean-100")
SAMPLE_AUDIO_DIR = LIBRISPEECH_ROOT / "200/126784"
SAMPLE_AUDIO_FILE = SAMPLE_AUDIO_DIR / "200-126784-0006.flac"

# Output paths
OUTPUT_DIR = PROJECT_ROOT / "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)  # Create output directory if it doesn't exist

# Default parameters
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_GAP_START_TIME = 2.0
DEFAULT_GAP_DURATION = 0.5

# Processing options
SUPPORTED_FORMATS = [".flac", ".wav", ".mp3"]