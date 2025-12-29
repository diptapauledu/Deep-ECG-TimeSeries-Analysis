import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt


def bandpass_filter(signal, lowcut=0.5, highcut=40.0, fs=360, order=4):
    if signal is None or len(signal) < 5:
        return signal

    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, signal)


def normalize_signal(signal):
    if signal is None or len(signal) == 0:
        return signal
    std = np.std(signal)
    if std == 0:
        return signal
    return (signal - np.mean(signal)) / std


def sliding_window(signal, window_size=3, step_size=1):
    windows = []
    if signal is None or len(signal) < window_size:
        return np.array(windows)

    for i in range(0, len(signal) - window_size + 1, step_size):
        windows.append(signal[i:i + window_size])

    return np.array(windows)


# ✅ DO NOT TOUCH INDENTATION HERE
def preprocess_ecg(csv_path):
    data = pd.read_csv(csv_path)

    ecg_signal = pd.to_numeric(
        data.iloc[:, 0], errors="coerce"
    ).dropna().values

    if len(ecg_signal) < 3:
        print("WARNING: ECG signal very short, continuing for demo.")

    filtered = bandpass_filter(ecg_signal)
    normalized = normalize_signal(filtered)
    segments = sliding_window(normalized)

    if len(segments) == 0:
        raise ValueError("No sliding windows could be created.")

    return segments
