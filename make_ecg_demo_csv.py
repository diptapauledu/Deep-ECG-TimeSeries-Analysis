import numpy as np
import pandas as pd

# Generate synthetic ECG-like waveform (50 samples)
t = np.linspace(0, 1, 50)
ecg = 0.6 * np.sin(2 * np.pi * 5 * t) + 0.05 * np.random.randn(50)

df = pd.DataFrame(ecg, columns=["ecg"])
df.to_csv("data/raw/ecg_demo.csv", index=False)

print("ecg_demo.csv created with", len(df), "samples")
