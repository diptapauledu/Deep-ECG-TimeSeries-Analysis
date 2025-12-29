import wfdb
import pandas as pd
import os

# Folder ensure
os.makedirs("data/raw", exist_ok=True)

# Load MIT-BIH record (record 100 is standard)
record = wfdb.rdrecord("100", pn_dir="mitdb")

# Take first ECG channel
ecg_signal = record.p_signal[:, 0]

# Convert to DataFrame
df = pd.DataFrame(ecg_signal, columns=["ecg"])

# Save as CSV
csv_path = "data/raw/ecg.csv"
df.to_csv(csv_path, index=False)

print("ECG CSV created successfully!")
print("Total samples:", len(df))
print("Saved at:", csv_path)
