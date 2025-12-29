import numpy as np
import pandas as pd

# VERY IMPORTANT:
# Manually download 100.dat and 100.hea
# Place them in the same folder as this script

def read_dat_file(filename):
    # MIT-BIH .dat is 16-bit signed integers
    data = np.fromfile(filename, dtype=np.int16)
    return data

signal = read_dat_file("100.dat")

df = pd.DataFrame(signal, columns=["ecg"])
df.to_csv("data/raw/ecg.csv", index=False)

print("ECG CSV created (offline method)")
print("Samples:", len(df))
