import numpy as np
import pandas as pd

# Load ECG demo data (50 samples)
data = pd.read_csv("data/raw/ecg_demo.csv")
ecg = pd.to_numeric(data.iloc[:, 0], errors="coerce").dropna().values

# Safety check
if len(ecg) < 20:
    raise ValueError("ECG data too short for saving predictions")

# Simulated prediction (slightly noisy version)
y_true = ecg
y_pred = ecg + 0.05 * np.random.randn(len(ecg))

# Save to txt files
np.savetxt("y_true.txt", y_true)
np.savetxt("y_pred.txt", y_pred)

print("y_true.txt and y_pred.txt successfully created")
print("Total samples:", len(y_true))
