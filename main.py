# =========================
# main.py
# =========================

print("DEBUG: main.py loaded")

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from src.train import train_model


# =========================
# FIGURE 2: Training Loss
# =========================
def plot_training_loss():
    loss_file = "results/loss_history.txt"

    if not os.path.exists(loss_file):
        print("WARNING: loss_history.txt not found, skipping Fig. 2")
        return

    with open(loss_file, "r") as f:
        losses = [float(x.strip()) for x in f.readlines()]

    if len(losses) < 2:
        print("WARNING: Not enough loss values for Fig. 2")
        return

    epochs = range(1, len(losses) + 1)

    os.makedirs("paper/figures", exist_ok=True)

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, losses, marker="o", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss (MSE)")
    plt.title("Training Loss Convergence")
    plt.grid(True)
    plt.tight_layout()

    plt.savefig("paper/figures/training_loss.png", dpi=300)
    plt.close()

    print("Fig. 2 saved: paper/figures/training_loss.png")


# =========================
# FIGURE 3: ECG DEMO (50 samples)
# =========================
def plot_ecg_demo(csv_path):
    os.makedirs("paper/figures", exist_ok=True)

    data = pd.read_csv(csv_path)
    ecg = pd.to_numeric(data.iloc[:, 0], errors="coerce").dropna().values

    if len(ecg) < 20:
        print("ERROR: ECG demo CSV must have at least 20 samples")
        return

    t = np.arange(len(ecg))

    plt.figure(figsize=(7, 4))
    plt.plot(t, ecg, linewidth=2, label="ECG Signal")

    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.title("ECG Signal Visualization")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig("paper/figures/ecg_prediction.png", dpi=300)
    plt.close()

    print("Fig. 3 saved: paper/figures/ecg_prediction.png")


# =========================
# MAIN PIPELINE
# =========================
def main():
    print("DEBUG: inside main")

    # ---- Train model (for loss curve) ----
    train_model(
        csv_path="data/raw/ecg_demo.csv",  # demo ECG with 50 samples
        epochs=10,
        lr=0.001
    )

    # ---- Generate figures ----
    plot_training_loss()                      # Fig. 2
    plot_ecg_demo("data/raw/ecg_demo.csv")   # Fig. 3

    print("PIPELINE DONE SUCCESSFULLY")


# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    main()
