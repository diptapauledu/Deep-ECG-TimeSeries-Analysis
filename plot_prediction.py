import numpy as np
import matplotlib.pyplot as plt

# Dummy example (replace with real prediction & ground truth)
t = np.linspace(0, 1, 200)
true_ecg = np.sin(2 * np.pi * 5 * t)
pred_ecg = true_ecg + 0.05 * np.random.randn(len(t))

plt.figure(figsize=(7, 4))
plt.plot(t, true_ecg, label="Original ECG", linewidth=2)
plt.plot(t, pred_ecg, label="Predicted ECG", linestyle="--")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig("results/ecg_prediction.png", dpi=300)
plt.show()
