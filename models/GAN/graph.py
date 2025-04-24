import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from pathlib import Path
import numpy as np

def smooth(values, weight=0.6):
    """Exponential moving average smoother."""
    smoothed = []
    last = values[0]
    for val in values:
        smoothed_val = last * weight + (1 - weight) * val
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

# === CONFIGURATION ===
log_paths = {
    "80ms": [
        Path("C:/Users/Nrew/Documents/VSCode Projects/ml-audio-inpainting/models/GAN/tensorboard/GAN-board_vgg_20250423_110015_80ms")
    ],
    "200ms": [
        Path("C:/Users/Nrew/Documents/VSCode Projects/ml-audio-inpainting/models/GAN/tensorboard/GAN-board_vgg_20250415_232312_200ms"),
        Path("C:/Users/Nrew/Documents/VSCode Projects/ml-audio-inpainting/models/GAN/tensorboard/GAN-board_vgg_20250416_233830_200ms")
    ]
}

tags = {
    "Generator_Adversarial": "Loss_Train/Generator_Adversarial",
    "Generator_Total": "Loss_Train/Generator_Total",
}

# === LOAD & PLOT ===
plt.figure(figsize=(12, 6))

for run_label, paths in log_paths.items():
    for tag_label, tag_name in tags.items():
        all_steps = []
        all_values = []

        for path in paths:
            ea = EventAccumulator(str(path))
            ea.Reload()
            try:
                scalars = ea.Scalars(tag_name)
                all_steps.extend([s.step for s in scalars])
                all_values.extend([s.value for s in scalars])
            except KeyError:
                print(f"Missing tag {tag_name} in {path.name}")

        # Sort the merged data in case steps reset during continuation
        sorted_pairs = sorted(zip(all_steps, all_values))
        sorted_steps, sorted_values = zip(*sorted_pairs)

        plt.plot(sorted_steps, sorted_values, label=f"{run_label} - {tag_label}")

plt.title("Generator Loss Comparison (80ms vs 200ms)")
plt.xlabel("Training Step")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("Generator_loss_comparison_merged.png", dpi=300)
