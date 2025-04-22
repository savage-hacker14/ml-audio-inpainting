<div align="center">
  <h1>Audio Inpainting</h1>
  <p><i>Recovering gapped audio with deep learning</i></p>

  <!-- PyTorch Badge -->
  <a href="https://pytorch.org/" target="_blank" style="text-decoration: none; display: inline-block;">
    <img src="https://img.shields.io/badge/PyTorch-2.0%2B-red?logo=pytorch&style=flat-square" />
  </a>

  <!-- License Badge -->
  <a href="https://opensource.org/licenses/MIT" target="_blank" style="text-decoration: none; display: inline-block;">
    <img src="https://img.shields.io/badge/License-MIT-blue?style=flat-square" />
  </a>

  <!-- Dataset Badge -->
  <a href="https://www.openslr.org/12" target="_blank" style="text-decoration: none; display: inline-block;">
    <img src="https://img.shields.io/badge/Dataset-LibraSpeech-8A2BE2?style=flat-square&logo=gitbook&logoColor=white&labelColor=gray" />
  </a>
</div>

---
This repository tackles **audio inpainting**, where damaged or missing parts of an audio signal are reconstructed by interpreting surrounding context. We use:

- **CNN + Bidirectional LSTM**
- **Generative Adversarial Network (GAN)**
- **Auto-Regressive models**

These models learn the distribution of natural audio and perform reconstruction in the **time-frequency domain** using spectrograms.

---

## Models

### 1. CNN + Bidirectional LSTM
- **Description**: Learns spatial and temporal audio features for gap recovery.
- **Directory**: `models/CNNBLSTM`
- **Configuration File**: `models/CNNBLSTM/config.yaml`

### 2. Generative Adversarial Network (GAN)
- **Description**: Combines a Generator and Discriminator with perceptual loss via VGG19.
- **Directory**: `models/GAN`
- **Configuration File**: `models/GAN/config.yaml`

### 3. Audio-Regressive
- **Description**: Predicts future frames based on past audio segments.
- **Directory**: `models/AudioReg`
- **Configuration File**: `models/AudioReg/config.yaml`

---

## Repository Structure
```
ml-audio-inpainting/
├── models/
│   ├── CNNBLSTM/       # CNN + BLSTM model
│   ├── GAN/            # GAN model with VGG loss
│   └── AudioReg/       # Auto-Regressive model
├── add_gaps.py         # Script to introduce gaps in audio
├── requirements.txt    # Dependencies
```

---

## Getting Started

### Prerequisitesa
1. Install Python (>= 3.8).
2. Install required dependencies using `pip`:
   ```bash
   pip install -r requirements.txt
3. Download the [LibraSpeach](https://www.openslr.org/12) dataset.

### How to Run a Model
Each model has its own directory and configuration file. General steps:


Each model has its dedicated directory and configuration file. Below are the steps to run the included models:

1. Navigate to the model's directory:
   ```bash
   cd models/{MODEL_NAME}
   ```

2. Edit the configuration file:
   ```bash
   vim config.yaml
   ```

3. Run the training script:
   ```bash
   python train.py
   ```
4. Run the testing script:
   ```bash
   python test.py
   ```
5. View results in:
   - `samples/`
   - `logs/`
   - `tensorboard/`
---

