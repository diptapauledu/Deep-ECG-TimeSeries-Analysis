import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, acf, pacf


def stationarity_test(signal):
    """
    Augmented Dickey-Fuller test for stationarity.
    """
    result = adfuller(signal)
    return {
        "ADF Statistic": result[0],
        "p-value": result[1],
        "Critical Values": result[4]
    }


def autocorrelation_analysis(signal, lags=40):
    """
    Compute ACF and PACF for time series analysis.
    """
    acf_vals = acf(signal, nlags=lags)
    pacf_vals = pacf(signal, nlags=lags)
    return acf_vals, pacf_vals


def plot_acf_pacf(signal, lags=40, save_path=None):
    """
    Plot ACF and PACF for visualization.
    """
    acf_vals, pacf_vals = autocorrelation_analysis(signal, lags)

    plt.figure()
    plt.stem(acf_vals)
    plt.title("Autocorrelation Function (ACF)")
    plt.xlabel("Lag")
    plt.ylabel("Correlation")
    if save_path:
        plt.savefig(save_path + "_acf.png")

    plt.figure()
    plt.stem(pacf_vals)
    plt.title("Partial Autocorrelation Function (PACF)")
    plt.xlabel("Lag")
    plt.ylabel("Correlation")
    if save_path:
        plt.savefig(save_path + "_pacf.png")

    plt.show()


if __name__ == "__main__":
    import pandas as pd
    data = pd.read_csv("data/raw/ecg.csv")
    ecg_signal = data.iloc[:, 0].values

    stat_result = stationarity_test(ecg_signal)
    print("Stationarity Test:", stat_result)

    plot_acf_pacf(ecg_signal, save_path="experiments/ecg")
